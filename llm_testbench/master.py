import subprocess
import time
import socket
import json
import os
import urllib.request

def run_hf_test():
    print("=== Starting Hugging Face benchmark")
    subprocess.run(["python", "hf_worker.py"], check=True)
    print("Hugging face finished, cooldown for 5 seconds...")
    time.sleep(5)

def is_server_ready(port: int) -> bool:
    try:
        # Pinging the official models route ensures the engine is 100% active
        url = f"http://localhost:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except Exception:
        return False
        
def wait_for_server(port: int, timeout: int = 600):
    print(f"Waiting for engine on port {port} to initialize...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_server_ready(port):
            print(f"Server on port {port} is ready \n")
            return True
        time.sleep(2)
    raise TimeoutError(f"Engine on port {port} failed to initialize within {timeout} seconds")


def run_vLLM_test():
    print("=== Starting vLLM benchmark===")

    vllm_cmd = [
        "docker", "run", "-d",
        "--name", "vllm-qwen",
        "--gpus", "all",
        "--ipc=host",
        "-e", "VLLM_USE_V1=0",
        "-p", "8000:8000",
        "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "vllm/vllm-openai:latest",
        "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.75",
        "--max-model-len", "1024",
        "--max-num-seqs", "16",

    ]

    print("Launching vLLM Docker container...")
    subprocess.run(vllm_cmd, check=True)

    try:
        wait_for_server(port=8000)
        print("Running vLLM client program...")
        subprocess.run(["python", "vllm_client.py"], check=True)
    
    finally:
        print("Cleaning up Docker container...")
        subprocess.run(["docker", "stop", "vllm-qwen"], check=False, stdout=subprocess.DEVNULL)          ####keywords
        subprocess.run(["docker", "rm", "vllm-qwen"], check=False, stdout=subprocess.DEVNULL)

        print("Cooldown for 5 seconds...")
        time.sleep(5)


def run_SGLang_test():
    print("=== Starting SGLang benchmark===")
    sglang_cmd = [
        "docker", "run", "-d",
        "--name", "sglang-qwen",
        "--gpus", "all",
        "--ipc=host",
        "--shm-size=16g",
        "-p", "30000:30000",
        "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "lmsysorg/sglang:latest",
        "sglang", "serve",  # Clean modern wrapper entry point
        "--model-path", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq_marlin",
        "--mem-fraction-static", "0.75",
        "--context-length", "1024",
        "--cuda-graph-max-bs-decode", "16",
        "--max-running-requests", "16", 
        "--port", "30000",
        "--host", "0.0.0.0"
    ]

    print("Launching SGLang Docker container...")
    subprocess.run(sglang_cmd, check=True)

    try:
        wait_for_server(port=30000)
        print("Running SGLang client program...")
        subprocess.run(["python", "sglang_client.py"], check=True)
    
    finally:
        print("Cleaning up Docker container...")
        subprocess.run(["docker", "stop", "sglang-qwen"], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", "sglang-qwen"], check=False, stdout=subprocess.DEVNULL)

        print("Cooldown for 5 seconds...")
        time.sleep(5)


def print_final_comparison_report():
    print("\n" + "="*85)
    print(f"{'Framework':<15} | {'Avg TTFT (s)':<13} | {'Decode (T/s)':<13} | {'Peak VRAM (GB)':<15} | {'Max GPU (%)':<11}")
    print("-" * 85)
    
    for backend, filename in [("Hugging Face", "hf_res.json"), ("vLLM", "vllm_res.json"), ("SGLang", "sglang_res.json")]:
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                
                # Normalizing keys depending on how you formatted your hf_worker logs
                ttft = data.get("avg_ttft", data.get("avg_ttft_sec", data.get("ttft", 0)))
                tps = data.get("avg_decode_tokens_per_sec", data.get("tokens_per_second", 0))
                vram = data.get("peak_vram_gb", data.get("max_mem", 0))
                gpu = data.get("max_gpu_compute_percent", data.get("max_gpu", 0))
                
                print(f"{backend:<15} | {ttft:<13.3f} | {tps:<13.2f} | {vram:<15.2f} | {gpu:<11.1f}")
            except FileNotFoundError:
                print(f"{backend:<15} | Metrics file missing!")
            
    print("="*85 + "\n")


def main():
    run_hf_test()
    
    run_vLLM_test()

    run_SGLang_test()

    print_final_comparison_report()

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()



