import requests
import json

url = "http://127.0.0.1:8000/generate"

payload = {
    "prompt": 'Hello',
    "max_new_tokens": 850,
    "temperature": 0.2
}

headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    print("Response Received:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Error: {response.status_code}")
    print(response.text)