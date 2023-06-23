import os
import re
import sys
import fitz  # PyMuPDF

def is_searchable_pdf(file_path):
    """
    Check if the PDF file is searchable.
    """
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            if page.search_for(" "):  # check for the presence of text
                return True
    return False

def extract_text_from_pdf(file_path, output_path):
    """
    Extract text from a searchable PDF and save it to a text file.
    """
    with fitz.open(file_path) as pdf:
        text = ""
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(text)

if __name__ == "__main__":
    source_directory = "C:\\python\\autoindex\\documents"
    output_directory = "C:\\python\\autoindex\\txt_output"

    os.makedirs(output_directory, exist_ok=True)

    extracted_files = []
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            input_file_path = os.path.join(root, file)
            
            if is_searchable_pdf(input_file_path):
                output_file_path = os.path.join(
                    output_directory, os.path.splitext(file)[0] + ".txt"
                )
                extract_text_from_pdf(input_file_path, output_file_path)
                print(f"Text extracted from {input_file_path} and saved as {output_file_path}.")
                extracted_files.append(input_file_path)
            else:
                print(f"{input_file_path} is not a searchable PDF.")
    
    # Print the list of extracted_files for main.py to read
    for extracted_file in extracted_files:
        print(extracted_file)