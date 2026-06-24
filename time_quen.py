from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
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
            self.start_time = time.time()
        

        elif self.is_first_token:
            self.ttft = time.time() - self.start_time
            self.sec_time = time.time()
            print("time:------>>>>", self.ttft)

            self.is_first_token = False
            self.start_time = time.time()

        elif self.is_second_token:
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

streamer.start_time = time.time()

start_time = time.time()
model.generate(
    **model_inputs,              #generate embeddings from token --> output response token??
    max_new_tokens=100,
    streamer=streamer,
)

end_time = time.time()
print("Total exec time >>>", end_time - start_time )
