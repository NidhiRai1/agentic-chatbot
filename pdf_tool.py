# pdf_tool.py

from fpdf import FPDF
import os
from datetime import datetime

def generate_pdf_report(content: str) -> str:
    """
    Generate a PDF file from the given content and return the full path to the file.
    This path will be used by Streamlit to trigger a file download.
    """
    # Ensure the output directory exists
    os.makedirs("pdfs", exist_ok=True)

    # Generate unique filename with timestamp
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join("pdfs", filename)

    # Create PDF content
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)

    # Save the file
    pdf.output(filepath)

    return filepath  # ✅ This will be used by Streamlit to show a download button
