import requests
import numpy as np
from pathlib import Path

OLLAMA = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a given text using the OLLAMA API.

    Args:
        text (str): The input text to be embedded."""
    response = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBEDDING_MODEL, "input": text})
    return np.array(response.json()["embeddings"])

def chat_with_model(prompt: str) -> str:
    """
    Get a response from the chat model using the OLLAMA API.

    Args:
        prompt (str): The input prompt for the chat model.  """

    response = requests.post(f"{OLLAMA}/api/chat", 
            json={"model": CHAT_MODEL, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            })
    return response.json()["message"]["content"]

# print(chat_with_model("What is the capital of France?"))
# print(get_embedding("What is the capital of France?"))

# Chunking and embedding a document

chunks = []

for file in Path("docs").glob("*.md"):
    chunks += [p.strip() for p in file.read_text().split("\n\n") if p.strip()]

chunk_embeddings = get_embedding(chunks)
print(f"Indexed {len(chunks)} chunks with embeddings of shape {chunk_embeddings.shape}.")

# retreiving relevant chunks based on a query

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    query_embedding = get_embedding(query)[0] 
    scores = chunk_embeddings @ query_embedding / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding))
    return [chunks[i] for i in np.argsort(scores)[::-1][:top_k]]

while True:
    query = input("Enter your query: ")
    context = "\n\n".join(retrieve_relevant_chunks(query))
    prompt = f"""Answer the following question using only the context below. If the answer is not in the context, say I don't know."

    Context: {context}

    Question: {query}"""
    answer = chat_with_model(prompt)
    print(f"Answer: {answer}")

