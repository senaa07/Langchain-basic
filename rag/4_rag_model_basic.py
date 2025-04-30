import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "sample.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vecotr store.")

    if not os.path.exists(file_path):
        raise FileExistsError(
            f"The file {file_path} does not exist.Please check the path."
        )

    loader = TextLoader(file_path)
    documents = loader.load()

    """
    documents.metadata = {"source": file_path}
    """
    text_spillter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_spillter.split_documents(documents)

    print("------ Document Chunks Information")
    print(f"Number of document Chunks: {len(docs)}")
    print(f"Sample chunk:\n {docs[0].page_content}\n")

    print("\n --creating Embeddings----")
    huggingFace_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    embeddings = OpenAIEmbeddings(model="text-embeddings-3-small")

    print("\n--- FInished creating embeddings---")

    db = Chroma.from_documents(
        docs, huggingFace_embeddings, persist_directory=persistent_directory
    )

    print("\n ---- Finsished creating vector store -----")

else:
    print("Vector store already exists.")
