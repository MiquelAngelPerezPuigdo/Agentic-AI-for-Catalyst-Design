import fitz  # PyMuPDF
from pathlib import Path

def extract_pdf_text_clean(pdf_path: Path) -> str:
    """Extracts and strips whitespace for high-quality text context."""
    try:
        doc = fitz.open(str(pdf_path))
        text = [page.get_text() for page in doc]
        return "\n".join(t.strip() for t in text if t.strip())
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""