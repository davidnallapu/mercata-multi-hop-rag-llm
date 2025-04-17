import requests

try:
    response = requests.get("http://localhost:8080/v1")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error connecting to Weaviate: {e}") 