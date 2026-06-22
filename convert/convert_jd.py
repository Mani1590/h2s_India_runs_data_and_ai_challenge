from docx import Document
import sys

doc = Document("data/job_description.docx")

with open("data/job_description.txt", "w", encoding="utf-8") as f:
    for para in doc.paragraphs:
        f.write(para.text + "\n")

print("Converted job_description.docx -> job_description.txt")
