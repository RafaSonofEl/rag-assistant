import os
import openai
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec  # ✅ this is new

# Load .env
load_dotenv()

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Pinecone v3+ setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Settings
CHUNK_SIZE = 500  # characters

def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

def upload_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        chunks = chunk_text(text, CHUNK_SIZE)
        print(f"Uploading {len(chunks)} chunks from {filepath}")
        vectors = []
        for chunk in chunks:
            vector = {
                "id": str(uuid.uuid4()),
                "values": get_embedding(chunk),
                "metadata": {
                    "text": chunk,
                    "source": os.path.basename(filepath)
                }
            }
            vectors.append(vector)
        index.upsert(vectors=vectors)
        print(f"✅ Uploaded {len(vectors)} vectors.")

def main():
    docs_path = "/Users/raphael/Documents/RAG/rag-knowledge-assistant/documents"
    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            upload_file(os.path.join(docs_path, file))

if __name__ == "__main__":
    main()
