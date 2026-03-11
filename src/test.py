from pypdf import PdfReader

reader = PdfReader("data/sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

print("First 500 characters of the document:\n")
print(text[:500])