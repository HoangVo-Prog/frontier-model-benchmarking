from frontier_client import FrontierClient

client = FrontierClient(model="MODEL_ID")  # Choose a vision-capable model.
response = client.prompt("Describe this picture.", image="images/example.png")
print(response.text)
