import subprocess
import time
import socket
import json

def run_hf_test():
    print("=== Starting Hugging Face benchmark")
    subprocess.run(["python", "hf_worker.py"], check=True)
    print("Hugging face finished, cooldown for 5 seconds...")
    time.sleep(5)

def is_server_ready(port: int) ->bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:              ####keywords
        try:
            s.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, OSError):
            return False
        
def wait_for_server(port: int, timeout: int = 120):
    print(f"Waiting for engine on port {port} to initialize...")
    start_time = time.time()
    while time.time - start_time < timeout:
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
        "-p", "8000:8000",
        "-v", "~/.cache/huggingface:/root/.cache/huggingface",
        "vllm/vllm-openai:latest",
        "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--gpu-memory-utilization", "0.75",
        "--max-model-len", "2048",

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
        "-v", "~/.cache/huggingface:/root/.cache/huggingface",
        "lmsysorg/sglang:latest",
        "sglang", "serve",
        "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--quantization", "awq",
        "--mem-fraction-static", "0.75",
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





def main():
    run_hf_test()
    
    run_vLLM_test()

    run_SGLang_test()

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()



