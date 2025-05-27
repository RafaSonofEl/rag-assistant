import openai
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    # Step 1: Get embedding
    embedding_response = openai.Embedding.create(
        input=[payload.question],
        model="text-embedding-3-small"
    )
    user_embedding = embedding_response["data"][0]["embedding"]

    # Step 2: Query Pinecone
    results = index.query(
        vector=user_embedding,
        top_k=3,
        include_metadata=True
    )

    # Step 3: Build context
    context = "\n---\n".join([match["metadata"]["text"] for match in results["matches"]])

    # Step 4: Get GPT response
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer only using the context below."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {payload.question}"}
        ]
    )

    return {"answer": completion["choices"][0]["message"]["content"]}

from fastapi.responses import HTMLResponse

@app.get("/chat", response_class=HTMLResponse)
def get_chat_ui():
    with open("chat_ui.html", "r", encoding="utf-8") as f:
        return f.read()