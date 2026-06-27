from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',
                                             quantization_config=quantization_config,
).cuda()

tokens = tokenizer.encode("The red kite was", return_tensors="pt").cuda()

time_start = time.time()
output = model.generate(
    tokens, max_new_tokens=300, use_cache = True,
)
time_decode = time.time()
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
time_end = time.time()
print("time total:::---->", time_end - time_start )