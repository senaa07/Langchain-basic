from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)
print("System message added to chat history.")

while True:
    query = input("You: ")
    if query.lower == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    print(f"AI: {response}\n")
    chat_history.append(AIMessage(content=response))

print("------Message History------")
print(chat_history)
