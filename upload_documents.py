import os
import openai
import uuid
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

CHUNK_SIZE = 500

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
        print(f"\u2705 Uploaded {len(vectors)} vectors.")

def main():
    docs_path = "./documents"
    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            upload_file(os.path.join(docs_path, file))

if __name__ == "__main__":
    main()