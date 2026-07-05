import asyncio
import time
import aiohttp
import json
import threading
from pynvml import *

generation_event = threading.Event()
gpu_telemetry_records = []

PORT = 30000
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
OUTPUT_FILE = "sglang_res.json"

prompts_suite = ["What do you know about the weather in antarctica?"] * 50

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
            full_response_text = ""
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
                    content = delta.get("content", "")

                    if content and ttft is None:
                        ttft = time.time() - start_time
                    
                    if content:
                        full_response_text += content
                        total_tokens += 1

                if "usage" in chunk and chunk["usage"] is not None:
                    total_tokens = chunk["usage"].get("completion_tokens", 0)

            end_time = time.time()
            total_duration = end_time - start_time

            if total_tokens == 0:
                total_tokens = 100

            decode_duration = total_duration - (ttft if ttft else 0)
            if decode_duration <= 0: 
                decode_duration = 0.001

            calculated_tokens = max(1, int(len(full_response_text) / 4))
            if total_tokens <= 0 or total_tokens == 1: 
                total_tokens = calculated_tokens

            tokens_per_sec = total_tokens / decode_duration

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



async def main():
    global gpu_telemetry_records

    api_endpoint = f"http://localhost:{PORT}/v1/chat/completions"
    print(f"Sending initial warmup request to port {PORT}...")
    warmup_payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "stream": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_endpoint, json=warmup_payload) as resp:
                await resp.json()
        print("Backend warm-up complete. Core hardware graphs traced.")
    except Exception as e:
        print(f"Warmup skipped or engine busy: {e}")

    generation_event.set()
    monitor_thread = threading.Thread(target=nvml_telemetry_monitor)
    monitor_thread.start()

    print(f"Blasting {len(prompts_suite)} cocurrent requests to engine port {PORT}")
    
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                send_request(session, api_endpoint, MODEL_NAME, prompts_suite[i], client_id=i)
                for i in range(len(prompts_suite))
            ]
            client_metrics = await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Error during parallel generation loop: {e}")
        client_metrics = []

    generation_event.clear()
    monitor_thread.join()
    
    valid_clients = [m for m in client_metrics if m is not None]
    avg_ttft = sum(c["ttft"] for c in valid_clients)/ len(valid_clients) if valid_clients else 0
    avg_tps = sum(c["tokens_per_sec"] for c in valid_clients) / len(valid_clients) if valid_clients else 0

    if len(gpu_telemetry_records) > 0:
        peak_vram = max(row[1] for row in gpu_telemetry_records)
        max_compute_util = max(row[2] for row in gpu_telemetry_records)
        avg_power_draw = sum(row[3] for row in gpu_telemetry_records) / len(gpu_telemetry_records)
    else:
        peak_vram, max_compute_util, avg_power_draw = 0, 0, 0

    final_report = {
        "avg_ttft_sec": avg_ttft,
        "avg_decode_tokens_per_sec": avg_tps,
        "peak_vram_gb": peak_vram,
        "max_gpu_compute_percent": max_compute_util,
        "avg_power_watts": avg_power_draw
    }

    json.dump(final_report, open(OUTPUT_FILE, "w"), indent=4)
    print(f"\nBenchmark completed. Summary written to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())


