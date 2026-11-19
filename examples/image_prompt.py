from frontier_client import FrontierClient

# Replace MODEL_ID with an allowed, vision-capable model from `frontier models list`.
# Replace photo.png with your own local file.
client = FrontierClient(model="MODEL_ID")
response = client.prompt("Describe this picture.", image="photo.png")
print(response.text)
