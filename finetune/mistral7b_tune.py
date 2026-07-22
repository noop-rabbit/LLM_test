import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LEN = 4096

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(tokenizer.special_tokens_map)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map= {"":0},
    dtype=torch.bfloat16, 
)

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
model.print_trainable_parameters()
# print(tokenizer.chat_template)

i = tokenizer.chat_template.find('|trim')
print(repr(tokenizer.chat_template[i-80:i+40]))

old = '{{- " " + message["content"]|trim + eos_token}}'
new = '{{- " " }}{% generation %}{{- message["content"]|trim + eos_token }}{% endgeneration %}'

assert old in tokenizer.chat_template                     # precondition: match will succeed
tokenizer.chat_template = tokenizer.chat_template.replace(old, new)
assert "{% generation %}" in tokenizer.chat_template      # postcondition: edit took


row = {"messages": [
    {"role": "user", "content": "Schema: users(id, name, signup_year)\n\nHow many users?"},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM users;"},
    {"role": "user", "content": "Only those who signed up after 2020."},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE signup_year > 2020;"},
]}

out = tokenizer.apply_chat_template(
    row["messages"],
    return_assistant_tokens_mask=True,
    return_dict=True,
    tokenize=True,
)
mask = out["assistant_masks"]
ids = out["input_ids"]
# print(tokenizer.decode([i for i, m in zip(ids, mask) if m]))
# print("{% generation %}" in tokenizer.chat_template)

dataset = Dataset.from_list(raw_examples).map(format_chatml)