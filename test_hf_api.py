import requests
import os
import os
print("HF token present?", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))

# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
# headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_TOKEN', '')}"}
# data = {"inputs": "show some shoes under 500 dollars"}

# print("Sending request to HuggingFace API...")
# try:
#     response = requests.post(API_URL, headers=headers, json=data, timeout=30)
#     print("Status code:", response.status_code)
#     print("Response:", response.text)
# except Exception as e:
#     print(f"Exception occurred: {e}") 
from langchain_huggingface import HuggingFaceEndpoint
ep = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=32,
)

print(ep.invoke("What is the capital of France?"))