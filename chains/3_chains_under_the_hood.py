from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model="gpt-4")

prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a comedian who tells jokes about {topic}.")(
            "human", "Tell me {joke_count} joke."
        )
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_prompt = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_prompt], last=parse_output)

response = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(response)
