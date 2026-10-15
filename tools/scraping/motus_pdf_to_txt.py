import pdfplumber
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
pdf_path = project_root / "data" / "source_material" / "ceceilmotus.pdf"
txt_path = project_root / "data" / "lexicons" / "raw" / "output.txt"
txt_path.parent.mkdir(parents=True, exist_ok=True)

with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as out_file:
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            out_file.write(f"--- Page {page_num} ---\n")
            out_file.write(text)
            out_file.write("\n\n")

print(f"✅ Done! Extracted PDF text saved to: {txt_path}")
