from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gtp-3.5-turbo")

result = model.invoke("What is the capital of France?")
print("FUll result:")
print(result)
print("Content Only:")
print(result.content)
