import asyncio
import time
import aiohttp

# Generate a synthetic ~4000 token system prompt prefix
BASE_SYSTEM_PROMPT = "token system prompt prefix: " * 1000 

async def send_request(session, user_query, run_label):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0,
        "max_tokens": 15,
        "stream": True
    }
    
    start_time = time.time()
    ttft = None
    
    async with session.post("http://localhost:30000/v1/chat/completions", json=payload) as response:
        async for chunk in response.content:
            if ttft is None:
                ttft = time.time() - start_time
                print(f"[{run_label}] TTFT achieved: {ttft:.4f} seconds")
        
        total_time = time.time() - start_time
        print(f"[{run_label}] Complete request roundtrip: {total_time:.4f} seconds\n")

async def main():
    async with aiohttp.ClientSession() as session:
        # Run 1: Cold Start (SGLang must prefill the entire 4k context)
        print("Executing Run 1 (Cold Cache Context)...")
        await send_request(session, "Summarize the core premise in one sentence.", "RUN 1 - COLD")
        
        # Short cooldown to separate logs
        await asyncio.sleep(2)
        
        # Run 2: Hot Start (SGLang should look up the system prompt prefix in the Radix Tree)
        print("Executing Run 2 (Hot Cache Context - Shared Prefix)...")
        await send_request(session, "What is the ultimate conclusion of this document?", "RUN 2 - HOT")

if __name__ == "__main__":
    asyncio.run(main())