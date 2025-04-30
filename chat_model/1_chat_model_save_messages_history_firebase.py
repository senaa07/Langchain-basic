from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

print("Initializing firebase connection...")
client = firestore.Client(project="your-project-id")

chat_history = FirestoreChatMessageHistory(
    session_id="Chat_Session_1", collection="ss", client=client
)

print("Chathistory is initialized.")
print(f"current Chat history {chat_history.messages}")

model = ChatOpenAI(model="gpt-3.5-turbo")
print("Model is initialized.")

while True:
    human_input = input("You: ")
    if human_input == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}\n")
