from pypdf import PdfReader

# Load the PDF
reader = PdfReader("data/sample.pdf")

# Extract text from all pages
text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Split text into chunks
chunk_size = 500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Print results
print(f"Total chunks: {len(chunks)}\n")

print("First chunk:\n")
print(chunks[0])

print("\n" + "=" * 50 + "\n")

if len(chunks) > 1:
    print("Second chunk:\n")
    print(chunks[1])