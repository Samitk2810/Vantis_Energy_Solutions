import os
from langchain_community.embeddings import JinaEmbeddings
from langchain_chroma import Chroma
from rag.chunker import chunks
from dotenv import load_dotenv
load_dotenv()

embeddings = JinaEmbeddings(jina_api_key=os.getenv("JINA_API_KEY"), model_name="jina-embeddings-v5-omni-small")

def build_vector_store():
    """Build the vector store from the KB chunks."""
    return Chroma.from_documents(chunks, embeddings, persist_directory="vector_db")


def load_vector_store():
    """Load the vector store from disk."""
    return Chroma(persist_directory="vector_db", embedding_function=embeddings)