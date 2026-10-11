from rag.vector_store import load_vector_store

vector_store = load_vector_store()
TOP_K = 4


def get_context(query: str, k: int = TOP_K) -> str:
    """Return the top-k KB chunks as one context string for the LLM."""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


if __name__ == "__main__":
    print(get_context("Do you install in Kota?"))