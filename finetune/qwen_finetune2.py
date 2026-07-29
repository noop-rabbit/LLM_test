import random
from datetime import date, timedelta
from textwrap import dedent
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch.optim as optim
import bitsandbytes as bnb
import json
from datasets import load_dataset


### ---------------------------------------Data generation-------------------------------------- ###
random.seed(42)

SYSTEM_PROMPT = """You are an invoice extraction system. Extract invoice fields from the provided text.
Return strict JSON only using the required schema.
Do not include Markdown, explanations, or extra text. Use null for missing values."""
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 512


def generate_invoice_truth():

  mylist1 = ["Apple", "Banana", "Cherry", "Milk", "Mango", "Sugar"]
  mylist2 = ["Gov.", "Republic", "Ltd.", "Enterprise", "Corp.", "Inc."]
  mylist3 = ["USD", "GBP", "INR", "EUR"]
  days = [15, 30, 45]

  amount = round(random.uniform(1, 10000), 2)
  rate = random.randint(5, 20) / 100
  tax = round(amount * rate, 2)
  total = round(amount + tax, 2)

  date_i = date(2024, 3, 8) + timedelta(days=random.randint(1,1000))
  date_d = date_i + timedelta(days=random.choice(days))

  inv_num = random.choice(["INV", "REF", "BILL"]) + "-" + str(random.randint(1000, 99999))

  invoice = {
      "vendor_name" : random.choice(mylist1)+ " " + random.choice(mylist2),
      "invoice_number" :  inv_num,
      "invoice_date" : date_i.isoformat(),
      "due_date" : date_d.isoformat(),
      "currency" : random.choice(mylist3),
      "subtotal" : amount,
      "tax" : tax,
      "total" : total,
  }

  return invoice


def render_invoice_text(invoice):

  symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}

  vendor_name = invoice.get("vendor_name", "NA")
  invoice_number = invoice.get("invoice_number", "NA")
  invoice_date = date.fromisoformat(invoice["invoice_date"]).strftime("%B %d, %Y")
  due_date = date.fromisoformat(invoice["due_date"]).strftime("%B %d, %Y")

  text = dedent(f"""\
  {vendor_name}

  Invoice Number: {invoice_number}
  Invoice Date: {invoice_date}
  Due Date: {due_date}

  Subtotal: {symbols[invoice['currency']]}{invoice['subtotal']:,.2f}
  Tax: {symbols[invoice['currency']]}{invoice['tax']:,.2f}
  Total: {symbols[invoice['currency']]}{invoice['total']:,.2f}
  """)

  return text


def build_example(invoice):
  example = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_invoice_text(invoice)},
        {"role": "assistant", "content": json.dumps(invoice, indent=2)}
    ]
}
  return example



examples = []
for _ in range(1000):
    invoice = generate_invoice_truth()
    examples.append(build_example(invoice))

for ex in examples:
    assert ex["messages"][1]["content"] is not None
    assert isinstance(ex["messages"][2]["content"], str)

assert round(invoice["subtotal"] + invoice["tax"], 2) == invoice["total"]

with open("invoice_train.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

train_examples = examples[:900]
val_examples = examples[900:]


def save_jsonl(path, data):
  with open(path, "w", encoding="utf-8") as f:
    for ex in data:
      f.write(json.dumps(ex, ensure_ascii=False) + "\n")

save_jsonl("invoice_train.jsonl", train_examples)
save_jsonl("invoice_val.jsonl", val_examples)

required_keys = {
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "currency", "subtotal", "tax", "total"
}

with open("invoice_train.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)

        assert "messages" in ex
        assert len(ex["messages"]) == 3
        assert ex["messages"][1]["content"] is not None

        assistant_json = json.loads(ex["messages"][2]["content"])
        assert set(assistant_json.keys()) == required_keys

print("train file looks good")

dataset = load_dataset(
    "json",
    data_files={
        "train": "invoice_train.jsonl",
        "validation":"invoice_val.jsonl",
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format_example(example):
  formatted_text = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
  )

  return {"text": formatted_text}

formatted_dataset = dataset.map(format_example)

##---------------------------- Training --------------------------------------##

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"":0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj","up_proj", "down_proj",
                    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

sft_config = SFTConfig(
    output_dir="Qwen_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=25,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    fp16=False,
    max_length=MAX_SEQ_LEN,
    assistant_only_loss=True,
    optim="paged_adamw_8bit",

)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["validation"],
    processing_class=tokenizer

)

trainer.train()