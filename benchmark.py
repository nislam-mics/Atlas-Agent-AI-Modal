import httpx
import time
import json
import uuid

def run_benchmark(query):
    url = "https://aidan-thomas--atlas-agent-ai-atlasagent-chat.modal.run"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "42b04e46a194dc4b573270f5517edac7"
    }
    payload = {
        "user_id": f"alex_bench_{uuid.uuid4().hex[:8]}",
        "message": query
    }
    
    # We follow redirects and handle the async Modal bridge
    start_time = time.time()
    try:
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            response = client.post(url, headers=headers, json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                latency = end_time - start_time
                print(f"Query: {query}")
                print(f"Latency: {latency:.2f}s")
                print(f"Response: {result.get('response')[:200]}...")
                print("-" * 30)
                return latency
            else:
                print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    return None

if __name__ == "__main__":
    queries = [
        "I need help with my housing application. I am very stressed.",
        "What is the EU AI Act Article 14 about?",
        "Where can I find the official documentation for the ATLAS project?"
    ]
    
    print("Starting Benchmarks...\n")
    for q in queries:
        run_benchmark(q)
