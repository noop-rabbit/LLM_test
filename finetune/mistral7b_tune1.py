import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch.optim as optim
import bitsandbytes as bnb

raw_examples = [
    {
        "schema": "users(id, name, signup_year)",
        "turns": [
            {"q": "How many users?", "a": "SELECT COUNT(*) FROM users;"},
            {
                "q": "Only those who signed up after 2020.",
                "a": "SELECT COUNT(*) FROM users WHERE signup_year > 2020;",
            },
            {
                "q": "Only those who signed up in 2022.",
                "a": "SELECT COUNT(*) FROM users WHERE signup_year = 2022;",
            },
            {
                "q": "Only those who signed up between 2021 and 2023.",
                "a": "SELECT COUNT(*) FROM users WHERE signup_year BETWEEN 2021 AND 2023;",
            },
        ],
    },

    {
        "schema": "employees(id, name, department, salary)",
        "turns": [
            {"q": "List all employees.", "a": "SELECT * FROM employees;"},
            {
                "q": "Only Engineering.",
                "a": "SELECT * FROM employees WHERE department = 'Engineering';",
            },
            {
                "q": "Only those earning more than 100000.",
                "a": "SELECT * FROM employees WHERE department = 'Engineering' AND salary > 100000;",
            },
            {
                "q": "Sort by salary descending.",
                "a": "SELECT * FROM employees WHERE department = 'Engineering' AND salary > 100000 ORDER BY salary DESC;",
            },
        ],
    },

    {
        "schema": "products(id, name, category, price, stock)",
        "turns": [
            {"q": "Show all products.", "a": "SELECT * FROM products;"},
            {
                "q": "Only Electronics.",
                "a": "SELECT * FROM products WHERE category = 'Electronics';",
            },
            {
                "q": "Only those under 500.",
                "a": "SELECT * FROM products WHERE category = 'Electronics' AND price < 500;",
            },
            {
                "q": "Only those in stock.",
                "a": "SELECT * FROM products WHERE category = 'Electronics' AND price < 500 AND stock > 0;",
            },
        ],
    },

    {
        "schema": "orders(order_id, customer_id, order_date, amount)",
        "turns": [
            {"q": "How many orders?", "a": "SELECT COUNT(*) FROM orders;"},
            {
                "q": "Only in 2024.",
                "a": "SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = 2024;",
            },
            {
                "q": "Only above 1000.",
                "a": "SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = 2024 AND amount > 1000;",
            },
            {
                "q": "Show the total amount instead.",
                "a": "SELECT SUM(amount) FROM orders WHERE YEAR(order_date) = 2024 AND amount > 1000;",
            },
        ],
    },

    {
        "schema": "students(id, name, major, gpa)",
        "turns": [
            {"q": "Show all students.", "a": "SELECT * FROM students;"},
            {
                "q": "Only Computer Science majors.",
                "a": "SELECT * FROM students WHERE major = 'Computer Science';",
            },
            {
                "q": "Only those with GPA above 3.5.",
                "a": "SELECT * FROM students WHERE major = 'Computer Science' AND gpa > 3.5;",
            },
            {
                "q": "Sort by GPA.",
                "a": "SELECT * FROM students WHERE major = 'Computer Science' AND gpa > 3.5 ORDER BY gpa ASC;",
            },
        ],
    },

    {
        "schema": "movies(id, title, genre, rating, release_year)",
        "turns": [
            {"q": "Show all movies.", "a": "SELECT * FROM movies;"},
            {
                "q": "Only Action movies.",
                "a": "SELECT * FROM movies WHERE genre = 'Action';",
            },
            {
                "q": "Rated above 8.",
                "a": "SELECT * FROM movies WHERE genre = 'Action' AND rating > 8;",
            },
            {
                "q": "Released after 2015.",
                "a": "SELECT * FROM movies WHERE genre = 'Action' AND rating > 8 AND release_year > 2015;",
            },
        ],
    },

    {
        "schema": "sales(id, region, year, revenue)",
        "turns": [
            {"q": "Total revenue?", "a": "SELECT SUM(revenue) FROM sales;"},
            {
                "q": "Only North region.",
                "a": "SELECT SUM(revenue) FROM sales WHERE region = 'North';",
            },
            {
                "q": "Only for 2023.",
                "a": "SELECT SUM(revenue) FROM sales WHERE region = 'North' AND year = 2023;",
            },
            {
                "q": "Average revenue instead.",
                "a": "SELECT AVG(revenue) FROM sales WHERE region = 'North' AND year = 2023;",
            },
        ],
    },
]


def format_for_mistral(example, turns_key="turns", schema_key="schema"):
  message = []
  for key in example[turns_key]:
    if message ==[]:
      message.append({"role": "user", "content": f"{example[schema_key]}\n\n{key['q']}"})
      message.append({"role": "assistant", "content": key["a"]})
    else:
      message.append({"role": "user", "content": key["q"]})
      message.append({"role": "assistant", "content": key["a"]})
  return {"messages": message}


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LEN = 2048

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(tokenizer.special_tokens_map)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map= {"":0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model.config.pad_token_id = tokenizer.pad_token_id

print("----->",model.get_memory_footprint() / (1024**3))
print(model.model.layers[0].self_attn.q_proj.weight.dtype)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

model.config.use_cache = False

print(torch.cuda.max_memory_allocated() / (1024**3))

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


model = get_peft_model(model, lora_config)

# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()

print(torch.cuda.max_memory_allocated() / (1024**3))
model.print_trainable_parameters()


old = '{{- " " + message["content"]|trim + eos_token}}'
new = '{{- " " }}{% generation %}{{- message["content"]|trim + eos_token }}{% endgeneration %}'

assert old in tokenizer.chat_template                     # precondition: match will succeed
tokenizer.chat_template = tokenizer.chat_template.replace(old, new)
assert "{% generation %}" in tokenizer.chat_template      # postcondition: edit took



# ---------------------------------------------  Test code for stress test ----------------------------------------------------------#

 
print(torch.cuda.max_memory_allocated() / (1024**3))

fake_batch =  torch.randint(low=0, high=tokenizer.vocab_size, size=(1,MAX_SEQ_LEN), device="cuda")
attention_mask = torch.ones_like(fake_batch)

parameters = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.AdamW(parameters, lr=2e-4)
optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=2e-4)
model.train()                      # ← engage training mode (checkpointing needs this)
print(model.training)              # → True
print(model.is_gradient_checkpointing)  # → True  (flag AND mode both required)
torch.cuda.reset_peak_memory_stats()


for step in range(17):
  outputs = model(input_ids=fake_batch, attention_mask=attention_mask,labels=fake_batch)
  loss = outputs.loss
  loss.backward()
  if (step + 1) % 16 == 0:
    optimizer.step()
    optimizer.zero_grad()
  print(f"step {step}: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
  m_vram = torch.cuda.max_memory_allocated()
print(m_vram / (1024 ** 3))


#------------------------------------- Trainer -------------------------------#
sft_config = SFTConfig(
    output_dir="./qlora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=5,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    max_length=MAX_SEQ_LEN,
    assistant_only_loss=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

dataset = Dataset.from_list(raw_examples).map(format_for_mistral)

split = dataset.train_test_split(test_size=0.05)

train_dataset=split["train"]
eval_dataset=split["test"]

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)


torch.cuda.reset_peak_memory_stats()
print(f"before training start: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
trainer.train()
print(f"After training: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
model.save_pretrained("./qlora_output/final_adapter")
tokenizer.save_pretrained("./qlora_output/final_adapter")
print("Adapter saved. Check size with: du -sh ./qlora_output/final_adapter")