import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "Qwen_output/checkpoint-285"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

test_invoice = """Orange Services

Invoice Number: TEST-99901
Invoice Date: March 14, 2027
Due Date: April 13, 2027

Subtotal: £1,250.50
Tax: £250.10
Total: £1,500.60
"""

messages = [
    {
        "role": "system",
        "content": "You are an invoice extraction system. Extract invoice fields from the provided text.\nReturn strict JSON only using the required schema. \nDo not include Markdown, explanations, or extra text. Use null for missing values."
    },
    {
        "role": "user",
        "content": test_invoice
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id= tokenizer.eos_token_id,
    )

new_tokens= outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(response)