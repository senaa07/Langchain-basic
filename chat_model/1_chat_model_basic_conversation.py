from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 multiplied by 5?"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
