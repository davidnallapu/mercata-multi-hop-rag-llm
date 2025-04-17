# import subprocess

# def clear_docker_volume(volume_name="weaviate_data"):
#     subprocess.run(["docker", "volume", "rm", volume_name], check=True)
#     print(f"Deleted volume: {volume_name}")

# clear_docker_volume()

from openai import OpenAI
import os, json

api_key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", api_key[:15], "...")      # should show sk-proj

# Inspect env
print("OPENAI_ORGANIZATION =", os.getenv("OPENAI_ORGANIZATION"))

client = OpenAI(api_key="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")
print("client.organization =", client.organization)   # must be None

# Make a simple query to the OpenAI API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what can you tell me about Python?"}
    ]
)

# Print the response
print("\nAPI Response:")
print(response.choices[0].message.content)
