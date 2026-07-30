from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype = torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base, "Qwen_output/checkpoint-285")
model = model.merge_and_unload()
model.save_pretrained("Qwen_output/merged", safe_serialization=True)

AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained("Qwen_output/merged")