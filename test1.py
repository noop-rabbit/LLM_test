import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2-9b-it"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    device_map="auto"
)

# 2. Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(model_id)


# 3. Format a prompt using the model's chat template
# messages = [{"role": "user", "content": "Write an essay about KV Cache"}]
messages = [
    #{"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Write an essay about KV Cache."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# 4. Tokenize inputs and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 5. Generate tokens
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)

# 6. Decode tokens back into text
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"LLM Response: {response}")