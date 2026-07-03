import asyncio
import time
import aiohttp
import json
import threading
from pynvml import *

generation_event = threading.Event()
gpu_telemetry_records = []

PORT = 8000
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
OUTPUT_FILE = "vllm_res.json"

prompts_suite = ["Short prompt"] * 50

async def send_request(session, url, model_name, prompt, client_id):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    start_time = time.time()
    ttft = None
    total_tokens = 0

    try:
        async with session.post(url, json=payload) as response:
            async for line_bytes in response.content:
                line = line_bytes.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content") and ttft is None:
                        ttft = time.time() - start_time

                if "usage" in chunk and chunk["usage"] is not None:
                    total_tokens = chunk["usage"].get("completion_tokens", 0)

            end_time = time.time()
            total_duration = end_time - start_time

            if total_tokens == 0:
                total_tokens = 100

            decode_duration = total_duration - (ttft if ttft else 0)
            tokens_per_sec = total_tokens / decode_duration if decode_duration > 0 else 0

            print(f"Client {client_id} -> TTFT: {ttft:.3f}s | Decode: {tokens_per_sec:.2f} T/s")
            return {"ttft": ttft, "tokens_per_sec": tokens_per_sec}
        
    except Exception as e:
        print(f"Client {client_id} Error: {e}")
        return None


def nvml_telemetry_monitor():
    global gpu_telemetry_records
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0) # Targeting primary GPU index 0
    
    while generation_event.is_set():
        sample_time = time.time()
        
        # Memory Info
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        used_mem = memory_info.used / (1024**3) # GB
        
        # Compute Utilization Rates
        utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        
        # Power Characteristics
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0 # Watts
        
        gpu_telemetry_records.append([sample_time, used_mem, gpu_util, power])
        time.sleep(0.005) # Aggressive high-frequency tracking (5ms)
        
    nvmlShutdown()



def main():
    pass


if __name__ == "__main__":
    main()


