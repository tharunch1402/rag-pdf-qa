from pypdf import PdfReader

reader = PdfReader("data/sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""

print("Pages:", len(reader.pages))
print(text[:1000])