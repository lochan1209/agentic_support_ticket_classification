import requests
import time

url = "http://127.0.0.1:8001/v1/generate"
payload = {"prompt": "Hello Hexaware Testing"}

print("--- Firing 7 fast concurrent requests ---")
for i in range(1, 8):
    response = requests.post(url, json=payload)
    print(f"Request {i} -> Status Code: {response.status_code} | Body: {response.json()}")