import os
import PyPDF4
import pytesseract
import warnings
import time
import sys
from pdf2image import convert_from_path
from PyPDF4.utils import PdfReadWarning
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Thread

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

warnings.filterwarnings('ignore', category=PdfReadWarning)

def is_searchable(pdf_path):
    with open(pdf_path, 'rb') as fr:
        reader = PyPDF4.PdfFileReader(fr)
        try:
            if reader.getPage(0).extractText():
                return True
        except:
            pass
    return False

def ocr_pdf(pdf_path):
    print(f"{time.ctime()}: Starting OCR on {os.path.basename(pdf_path)}")
    output_folder = r"C:\python\autoindex\txt_output"
    output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '_OCR.txt'
    output_path = os.path.join(output_folder, output_filename)
    images = convert_from_path(pdf_path)
    
    result = []
    for image in images:
        text = pytesseract.image_to_string(image)
        result.append(text)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in result:
            outfile.write(line)
    
    print(f"{time.ctime()}: Finished OCR on {os.path.basename(pdf_path)}")

class TimeoutException(Exception):
    pass

def timeout(seconds):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            res = [TimeoutException(f'Timed out after {seconds} seconds')]
            def target():
                res[0] = function(*args, **kwargs)
            t = Thread(target=target)
            t.daemon = True
            t.start()
            t.join(seconds)
            if isinstance(res[0], Exception):
                raise res[0]
            return res[0]
        return wrapper
    return decorator

@timeout(180)  # Set a timeout of 3 minutes (180 seconds) for each PDF
def process_pdf(file):
    pdf_path = os.path.join(input_folder, file)
    try:
        if not is_searchable(pdf_path):
            ocr_pdf(pdf_path)
    except PyPDF4.utils.PdfReadError:
        print(f"Warning: Could not read the malformed PDF file: {os.path.basename(pdf_path)}")
    except KeyError as e:
        print(f"Warning: KeyError '{e}' encountered while processing {os.path.basename(pdf_path)}. Skipping the file.")
    except Exception as e:
        print(f"Unexpected error: {str(e)} while processing {os.path.basename(pdf_path)}")

def main():
    global input_folder
    input_folder = r'C:\python\autoindex\documents'
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf') and not f.endswith('_OCR.pdf')]
    
    num_cores = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        try:
            executor.map(process_pdf, pdf_files)
        except Exception as e:
            print(f"Unexpected error: {str(e)}\n{sys.exc_info()}")

if __name__ == '__main__':
    main()