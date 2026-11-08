import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Load and Chunk Documents
print("📚 Loading knowledge base...")
with open("knowledge.txt", "r", encoding="utf-8") as f:
    # Split documents by line
    documents = [line.strip() for line in f if line.strip()]

# 2. Generate Embeddings for Knowledge Base
print("⚡ Generating vector embeddings for documents...")
doc_embeddings = []
for doc in documents:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc,
    )
    doc_embeddings.append(response.embeddings[0].values)

doc_embeddings = np.array(doc_embeddings)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n🔍 RAG System Ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break

    # 3. Semantic Search: Embed the user query
    query_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    query_embedding = np.array(query_response.embeddings[0].values)

    # Calculate similarities across all documents
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    best_match_idx = np.argmax(similarities)
    retrieved_context = documents[best_match_idx]
    similarity_score = similarities[best_match_idx]

    print(f"\n[Retrieved Context (Score: {similarity_score:.2f})]: {retrieved_context}")

    # 4. Augmented Generation with Citations
    prompt = f"""
    You are an expert AI assistant. Answer the user's question using ONLY the provided context below. 
    You must include a direct citation of the source document used in your answer.

    Context:
    {retrieved_context}

    Question: {query}
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )

    print(f"\nAI Answer:\n{response.text}\n{'-'*40}\n")
