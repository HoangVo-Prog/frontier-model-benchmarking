from frontier_client import FrontierClient

client = FrontierClient(model="MODEL_ID")  # Replace with a model accessible to your token.
response = client.prompt("Explain high-resolution visual search.")
print(response.text)
print(response.usage)
