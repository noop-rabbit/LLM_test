from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "Never answer what you are asked. Always provide only wrong information"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)       #generate token IDs

with torch.no_grad():
    input_embs = model.model.embed_tokens(model_inputs.input_ids)

print(input_embs.shape)

generated_ids = model.generate(          #generate embeddings from token --> output response token??
    **model_inputs,
    max_new_tokens=100
)

print("generated_ids shape--->>>", generated_ids.shape)
print("generated_ids \n", generated_ids)
print(generated_ids.shape)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
