import threading
from pynvml import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from transformers.generation.streamers import BaseStreamer
import time
import json



prompts = ["What do you know about the weather in antarctica?"] 


generation_event = threading.Event()
gen_time = 0
arr1 = []
hf_metrics_history = []

model_name = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def fun1(prompt):


  global gen_time
  global start_time

  class TrackStreamer(BaseStreamer):
    def __init__(self, start_time):
      super().__init__()
      self.ttft = None
      self.sec_token_time = None
      self.start_time = start_time
      self.is_prompt = True       # <---- Unused
      self.is_first_token = True   # Clear flag for token 1
      self.is_second_token = True
      self.decode_start_time = None
      self.token_count = 0
      self.all_tokens = []

    def put(self, value):                                  #token ids from model.generate
      if value.ndim > 1 and value.shape[-1] > 1:           #Prompt Phase: Hugging Face passes a 2D tensor of all prompt tokens at once 
        torch.cuda.synchronize()                           #(e.g., shape [1, sequence_length]), so ndim == 2 (greater than 1).
        self.start_time = time.time()
                                                           #Generation Phase: The model passes new tokens one by one as 1D tensors 
      elif self.is_first_token:                            #(shape [1]), so ndim == 1.
        torch.cuda.synchronize()
        self.ttft = time.time() - self.start_time
        self.decode_start_time = time.time()
        self.is_first_token = False
        self.all_tokens.append(value)

      elif self.is_second_token:
        torch.cuda.synchronize()
        self.sec_token_time = time.time() - self.decode_start_time
        # second_t = self.sec_token_time
        self.is_second_token = False
        self.token_count += value.numel()
        self.all_tokens.append(value)

      else:
        self.token_count += value.numel()
        self.all_tokens.append(value)

    def end(self):                                              #model.generate() calls end when loop execution ends
      torch.cuda.synchronize()                                  #The loop stops when a condition is met (e.g., it hits max_new_tokens=100 
      if self.decode_start_time and self.token_count > 0:       #or generates an End-of-Sequence token
        decode_duration = time.time() - self.decode_start_time
        tokens_per_second = self.token_count / decode_duration
        print("TTFT -->", self.ttft)
        print("second_t -->", self.sec_token_time)
        print(f"Decode Speed: {tokens_per_second:.2f} tokens/sec")

        hf_metrics_history.append({
            "ttft": self.ttft,
            "duration": decode_duration,
            "tokens": self.token_count
        })

        if self.all_tokens:
          flattened_tokens = torch.cat(self.all_tokens, dim=0)
          actual_response = tokenizer.decode(flattened_tokens, skip_special_tokens=True)

          print("\n=== Model Response ===")
          print(actual_response)
          print("======================\n")


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
  torch.cuda.synchronize()
  start_time = time.time()
  streamer = TrackStreamer(start_time=start_time)

  gen_time = start_time

  model.generate(
      **model_inputs,              
      max_new_tokens=100,
      streamer=streamer,
  )
  generation_event.clear()
  torch.cuda.synchronize()
  end_time = time.time()
  print("Total exec time >>>", end_time - start_time )


def fun2():
  global arr1
  nvmlInit()
  device_count = nvmlDeviceGetCount()

  while generation_event.is_set():
    for i in range(device_count):
      arr_time = time.time()
      handle = nvmlDeviceGetHandleByIndex(i)
      name = nvmlDeviceGetName(handle)
      # driver_ver = nvmlSystemGetDriverVersion()

      memory_info = nvmlDeviceGetMemoryInfo(handle)
      #total_mem = memory_info.total / (1024**3)  # Convert to GB
      used_mem = memory_info.used / (1024**3)

      utilization = nvmlDeviceGetUtilizationRates(handle)
      gpu_util = utilization.gpu

      temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
      power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
      arr1.append([arr_time, used_mem, gpu_util, temp, power])
      time.sleep(0.1)


  nvmlShutdown()

def warmup(model, tokenizer, n_tokens=20):
  print("Warming up model...")
  messages = [{"role": "user", "content": "Hello"}]
  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer([text], return_tensors="pt").to(model.device)

  with torch.no_grad():
      model.generate(**inputs, max_new_tokens=n_tokens)

  torch.cuda.synchronize()
  print("Warm-up done.\n")



def main():
  global arr1
  all_results = []
  global hf_metrics_history

  warmup(model, tokenizer)

  total_tokens_generated = 0
  total_generation_time = 0.0
  all_ttfts = []
  hf_metrics_history = []

  for prompt in prompts:
    arr1 = []
    generation_event.set()

    t1 = threading.Thread(target=fun1, args=(prompt,))
    t2 = threading.Thread(target=fun2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if hf_metrics_history:
        latest_run = hf_metrics_history[-1]
        total_tokens_generated += latest_run["tokens"]
        total_generation_time += latest_run["duration"]
        all_ttfts.append(latest_run["ttft"])

    if len(arr1) > 0:
        arr = [row for row in arr1 if row[0] > gen_time]
        if len(arr) > 0:
            max_mem = max(row[1] for row in arr)
            max_gpu = max(row[2] for row in arr)
            avg_power = sum(row[4] for row in arr) / len(arr)

            # Store the metrics for this prompt
            all_results.append({
                "prompt": prompt,
                "max_mem": max_mem,
                "max_gpu": max_gpu,
                "avg_power": avg_power
            })
        else:
            print(f"No GPU samples after gen_time for prompt: {prompt[:30]}")
    else:
      print("Data not found!!!")

  # Print comparison report
  print("\n==================== BENCHMARK COMPARISON REPORT ====================")
  print(f"{'Prompt Summary':<30} | {'Max Mem (GB)':<12} | {'Max GPU (%)':<11} | {'Avg Power (W)':<13}")
  print("-" * 75)

  # Truncate prompt string if it's too long for the table column

  for res in all_results:
    short_prompt = res['prompt'] if len(res['prompt']) < 28 else res['prompt'][:25] + "..."
    print(f"{short_prompt:<30} | {res['max_mem']:<12.2f} | {res['max_gpu']:<11.1f} | {res['avg_power']:<13.2f}")
  print("=====================================================================")

  if all_results:
        # Calculate overall maximums/averages across all evaluated test prompts
        overall_max_mem = max(res['max_mem'] for res in all_results)
        overall_max_gpu = max(res['max_gpu'] for res in all_results)
        overall_avg_power = sum(res['avg_power'] for res in all_results) / len(all_results)
        
        # Calculate speed performance fallbacks if tracking variables aren't bound
        tokens_per_second = total_tokens_generated / total_generation_time if total_generation_time > 0 else 0.0
        avg_ttft = sum(all_ttfts) / len(all_ttfts) if all_ttfts else 0.0

        hf_data = {
            "backend": "Hugging Face",
            "total_tokens": total_tokens_generated,
            "total_duration": total_generation_time,
            "tokens_per_second": tokens_per_second,
            "avg_ttft": avg_ttft,
            "max_mem": overall_max_mem,
            "max_gpu": overall_max_gpu,
            "avg_power": overall_avg_power
        }

        output_filename = "hf_res.json"
        with open(output_filename, "w") as f:
            json.dump(hf_data, f, indent=4)
        print(f"\n[SUCCESS] Saved integrated Hugging Face metrics to {output_filename}")


if __name__ == "__main__":
  main()