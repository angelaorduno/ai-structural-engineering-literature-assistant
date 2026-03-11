from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF
reader = PdfReader("data/sample.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Chunk text
chunk_size = 500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Create embeddings
embeddings = model.encode(chunks)

# Convert to numpy array of float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Ask a question
query = "What machine learning algorithms are used in structural health monitoring?"
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Search top 3 most relevant chunks
distances, indices = index.search(query_embedding, k=3)

print("Question:")
print(query)
print("\nTop matching chunks:\n")

for i, idx in enumerate(indices[0], start=1):
    print(f"Match {i}:")
    print(chunks[idx])
    print("\n" + "=" * 80 + "\n")