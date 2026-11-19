from frontier_client import FrontierClient

client = FrontierClient(model="MODEL_ID")
with client.chat(system_prompt="You are a helpful assistant.") as chat:
    chat.send("Explain transformers.")
    print(chat.send("Explain that more simply.").text)
    chat.export("my_conversation.json")
