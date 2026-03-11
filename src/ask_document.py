from dotenv import load_dotenv
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

load_dotenv()

print("API key loaded:", os.getenv("OPENAI_API_KEY") is not None)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

reader = PdfReader("data/sample.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

chunk_size = 500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

question = input("Ask a question about the document: ")

query_embedding = model.encode([question])
query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(query_embedding, k=3)

context = "\n\n".join([chunks[i] for i in indices[0]])

prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nAnswer:\n")
print(response.choices[0].message.content)