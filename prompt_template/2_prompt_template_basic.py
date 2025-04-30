from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

template = "Tell me a joke about {topic}."

prompt_template = ChatPromptTemplate.from_template(template)

print("-------prompt from Template ---------")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

## More with multiple topic
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""

prompt_template = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_template.invoke({"adjective": "funny", "animal": "panda"})

print("\n--- Prompt with multiple Placeholders -----\n")
print(prompt)

messages = [
    ("system", "you are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

print("\n---- Prompt with system and Human messages (Tuple) ----\n")
print(prompt)
