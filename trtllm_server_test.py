import asyncio
import time
import aiohttp
import json

PORT = 8000
model_name = "qwen2.5-1.5b"
url = "http://localhost:8000/v1/chat/completions"

prompt = """Antarctica is the coldest, windiest, and driest continent on Earth, presenting a climate that is completely unique in its severity. The massive landmass is almost entirely buried beneath a thick ice sheet that averages roughly two kilometers in depth, containing nearly ninety percent of the planet's total ice volume and approximately seventy percent of its available fresh water. Structurally, the continent can be divided into two distinct meteorological regions: the vast, elevated high-altitude interior plateau and the significantly more moderate coastal margins.  The interior plateau experiences an intense polar ice-cap climate. Due to its high geographical latitude and extreme average elevation of three thousand meters, solar radiation strikes the surface at a highly oblique angle. This spread-out sunlight imparts minimal thermal energy, a phenomenon heavily compounded by the high albedo effect of the unbroken snow, which reflects up to eighty-five percent of solar radiation back into space. Consequently, annual mean temperatures inland hover around minus fifty-five degrees Celsius. During the dark winter months, temperatures on the East Antarctic Ice Sheet regularly plummet below minus seventy degrees Celsius, with the absolute lowest recorded surface temperature reaching minus eighty-nine point two degrees Celsius at the remote Vostok Station.  In contrast, the coastal areas and the prominent Antarctic Peninsula feature a maritime polar climate. The surrounding Southern Ocean exerts a powerful moderating influence, preventing the extreme thermal drops seen inland. During the summer travel season from November to March, coastal temperatures regularly climb above freezing, frequently ranging between minus two and eight degrees Celsius, with peak anomalies occasionally surpassing fifteen degrees Celsius on the western side of the peninsula.  Despite being covered in frozen water, Antarctica is classified as a hyper-arid polar desert. The extreme cold limits the moisture-bearing capacity of the air, resulting in an absolute humidity close to zero. The interior plateau receives less than fifty millimeters of water-equivalent precipitation annually, often falling merely as minute airborne ice crystals known colloquially as diamond dust. The coastal fringes receive more substantial deposits, averaging two hundred to five hundred millimeters per year, primarily in the form of heavy snowfall driven by low-pressure cyclonic systems moving off the Southern Ocean.  Wind is the definitive dynamic factor of the region. The continent’s topographically sloped profile drives the formation of fierce gravity-fed katabatic winds. Radiative cooling over the elevated central ice sheet creates an immense body of exceptionally cold, dense air. This heavy air mass spills downward toward the coast, accelerating rapidly through narrow glacial valleys. These persistent winds regularly exceed speeds of one hundred kilometers per hour, producing severe, multi-day blizzards and absolute whiteout conditions that completely erase visual horizons by lifting loose surface snow into the upper atmosphere. """




async def send_request(session, url, model_name, prompt, client_id):

    token_timestamps = []
    gaps = []

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    start_time = time.monotonic()
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
                  
                    if content:
                        token_timestamps.append(time.monotonic() - start_time)
                        full_response_text += content
                        total_tokens += 1
                        
            ttft = token_timestamps[0]
                    
            for i in range(1, len(token_timestamps)):
                gap = token_timestamps[i] - token_timestamps[i-1]
                gaps.append(gap)

               
            return {"client_id": client_id,"ttft": ttft, "gaps": gaps, "total_tokens": total_tokens }
        
    except Exception as e:
        print(f"Client {client_id} Error: {e}")
        return None
    

async def run_concurrency_level(session, url, model_name, prompt, concurrency):
    
    tasks = [send_request(session, url, model_name, prompt, client_id=i) for i in range(concurrency)]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r is not None]

    ttfts = [r["ttft"] for r in successful]
    all_gaps = [gap for r in successful for gap in r["gaps"]]

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
    avg_gap = sum(all_gaps) / len(all_gaps) if all_gaps else None

    failure_rate = (len(results) - len(successful)) / len(results)

    return {
        "concurrency": concurrency,
        "avg_ttft": avg_ttft,
        "avg_inter_token_latency": avg_gap,
        "failure_rate": failure_rate,
        "num_requests": len(results),
    }
        

async def main():

    api_endpoint = f"http://localhost:{PORT}/v1/chat/completions"
    print(f"Sending initial warmup request to port {PORT}...")
    warmup_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "stream": False
    }
    final_result = []
    concurrency_list = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]

    try:
        connector = aiohttp.TCPConnector(
        limit=500,            # Change global simultaneous connection limit (default is 100)
        limit_per_host=500     # Limit concurrent connections to the SAME endpoint (default is 0/None)
    )
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(api_endpoint, json=warmup_payload) as resp:
                await resp.json()
        print("Backend warm-up complete. Core hardware graphs traced.")
    except Exception as e:
        print(f"Warmup skipped or engine busy: {e}")

    
    
    
    try:
        async with aiohttp.ClientSession() as session:
            for num in concurrency_list:
                print(f"Blasting {num} cocurrent requests to engine port {PORT}")
                result = await run_concurrency_level(session, url, model_name, prompt, concurrency=num)
                final_result.append(result)
            print(final_result)
    except Exception as e:
        print(f"Error during parallel generation loop: {e}")
        print(final_result)
        
    return final_result

if __name__ == "__main__":
    asyncio.run(main())

