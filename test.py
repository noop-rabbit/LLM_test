import aiohttp, asyncio, time

url = "http://localhost:8080/v1/completions"
prompt = ["Explain the weather of Antarctica"] * 50

async def send_request(client, prompt, client_id):
    start_time = time.time()
    payload = {
        "model": "/engines/qwen_1.5b_engine",
        "prompt": prompt,
        "max_tokens": 200,
        "stream": False,
        "ignore_eos": True
    }
    async with client.post(url, json=payload) as response:
        result = await response.json()
    latency = time.time() - start_time
    token_count = result["usage"]["completion_tokens"]
    return token_count, latency

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, prompt=prompt[i], client_id=i) for i in range(50)]
        batch_start = time.time()
        results = await asyncio.gather(*tasks)
        batch_end = time.time()

        total_tokens = sum(item[0] for item in results)
        batch_duration = batch_end - batch_start
        avg_latency = sum(item[1] for item in results) / 50
        throughput = total_tokens / batch_duration

        print(f"Total tokens: {total_tokens}")
        print(f"Batch duration: {batch_duration:.3f}s")
        print(f"Avg per-request latency: {avg_latency:.3f}s")
        print(f"Aggregate throughput: {throughput:.2f} tokens/sec")

asyncio.run(main())