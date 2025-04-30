import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_dir = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embeddings-3-small")

db = Chroma(persist_directory=persistant_dir, embedding_function=embeddings)

query = "WHo is odysseus' wife"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant DOcuments ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n {doc.page_content}\n")

    if doc.metadata:
        print(f"Source: {doc.metadata.get('source','unknown')}\n")
