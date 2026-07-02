'''
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
import time

model_name = "Qwen/Qwen2.5-7B-Instruct"

# 2. Force 4-bit quantization to shrink model size to ~5.5GB
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load model completely into GPU VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto" # This will now map 100% to 'cuda:0'
)

#print(model.hf_device_map)



tokenizer = AutoTokenizer.from_pretrained(model_name)

class TrackStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.start_time = None
        self.ttft = None
        self.sec_token_time = None
        self.is_prompt = True
        self.is_first_token = True   # Clear flag for token 1
        self.is_second_token = True

    def put(self, value):
        if self.is_prompt:
            self.is_prompt = False
            torch.cuda.synchronize()
            self.start_time = time.time()
        

        elif self.is_first_token:
            torch.cuda.synchronize()
            self.ttft = time.time() - self.start_time
            self.sec_time = time.time()
            print("time:------>>>>", self.ttft)

            self.is_first_token = False
            self.start_time = time.time()

        elif self.is_second_token:
            torch.cuda.synchronize()
            self.sec_token_time = time.time() - self.start_time
            print("second token time:------>>>>", self.sec_token_time)
            self.is_second_token = False


        super().put(value)

prompt = "Give me a short introduction to large language model."


messages = [
    {"role": "system", "content": "You are a helpful AI assistant who provides correct information, always"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)       #generate token IDs

streamer = TrackStreamer(tokenizer)

torch.cuda.synchronize()
streamer.start_time = time.time()
torch.cuda.synchronize()
start_time = time.time()
model.generate(     
    **model_inputs,              #generate embeddings from token --> output response token??
    max_new_tokens=100,
    streamer=streamer,
)
torch.cuda.synchronize()
end_time = time.time()
print("Total exec time >>>", end_time - start_time )
