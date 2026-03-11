from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF
reader = PdfReader("data/sample.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Chunk text
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print(f"Chunks created: {len(chunks)}")

# Create embeddings
embeddings = model.encode(chunks)

print("\nEmbedding vector shape:")
print(embeddings.shape)