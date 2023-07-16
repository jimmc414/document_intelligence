# Import libraries
import pytesseract
from PyPDF2 import PdfWriter, PdfReader
import io
import os
from pdf2image import convert_from_path
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Set paths for Tesseract and poppler
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path = r'C:\python\autoindex\poppler-0.68.0\bin'
doc_dir = 'c:\\python\\autoindex\\documents'

def convert_pdf_to_searchable(filename):
    # Convert scanned PDF to image object
    images = convert_from_path(os.path.join(doc_dir, filename), poppler_path=poppler_path)

    # Initialize PDF Writer
    pdf_writer = PdfWriter()

    # Convert image object to searchable PDF
    for image in images:
        page = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
        page_pdf = PdfReader(io.BytesIO(page))
        pdf_writer.add_page(page_pdf.pages[0])

    # Save the searchable PDF
    base_filename = os.path.splitext(filename)[0]  # Separate base from extension
    new_filename = base_filename + '_searchable.pdf'
    with open(os.path.join(doc_dir, new_filename), "wb") as f_out:
        pdf_writer.write(f_out)

    # Print finished message with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Finished {new_filename}.")

if __name__ == "__main__":
    start_time = time.time()

    # Get a list of all pdfs in the given directory
    pdf_files = [file for file in os.listdir(doc_dir) if file.lower().endswith('.pdf')]

    # Create a pool of processes
    with Pool(cpu_count()) as p:
        p.map(convert_pdf_to_searchable, pdf_files)

    end_time = time.time()

    # Print total elapsed time
    elapsed_time = end_time - start_time
    elapsed_time_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    print(f"Total elapsed time: {elapsed_time_formatted}")
