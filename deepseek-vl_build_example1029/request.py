import requests
import json

url = "http://localhost:8000/chat"
data = {
    "messages": [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>summerize this image beyond 100 words<|/ref|>.",
            "images": ["./images/visual_grounding_2.jpg"]
        }
    ],
    "max_new_tokens": 512,
    "do_sample": False,
    "use_cache": True
}

response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

if response.status_code == 200:
    result = response.json()
    print("Response:", result["response"])
else:
    print(f"Error: {response.status_code} - {response.text}")