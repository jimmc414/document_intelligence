import os
import cv2
import PyPDF4
import pytesseract
import warnings
import time
import sys
import numpy as np
from pdf2image import convert_from_path
from PyPDF4.utils import PdfReadWarning
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Thread
from scipy.ndimage import interpolation as inter
from PIL import Image

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

def deskew(image):
    image = np.array(image)
    image = image[:, :, ::-1].copy()  # Convert to BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(rotated[:, :, ::-1].copy())  # Convert back to RGB and return as PIL Image

def ocr_pdf(pdf_path):
    print(f"{time.ctime()}: starting OCR {os.path.basename(pdf_path)}")
    output_folder = r"c:\python\autoindex\txt_output"
    output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + "_ocr.txt"
    output_path = os.path.join(output_folder, output_filename)
    
    images = convert_from_path(pdf_path)
    result = []
    
    custom_config = r'–psm 6 --oem 1'
    for image in images:
        deskewed_image = deskew(image)
        text = pytesseract.image_to_string(deskewed_image, config=custom_config)
        result.append(text)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in result:
            outfile.write(line)

    print(f"{time.ctime()}: finished OCR {os.path.basename(pdf_path)}")

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

@timeout(1000)  # Set a timeout of 3 minutes (180 seconds) for each PDF
def process_pdf(file):
    pdf_path = os.path.join(input_folder, file)
    try:
        # if not is_searchable(pdf_path):
        ocr_pdf(pdf_path)
    except PyPDF4.utils.PdfReadError:
        logging.error(f"Could not read the malformed PDF file: {os.path.basename(pdf_path)}")
    except KeyError as e:
        logging.error(f"KeyError '{e}' encountered while processing {os.path.basename(pdf_path)}. Skipping the file.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)} while processing {os.path.basename(pdf_path)}")

def main():
    global input_folder
    input_folder = r'C:\python\autoindex\documents'
    pdf_files = sys.argv[1:]

    num_cores = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        try:
            executor.map(process_pdf, pdf_files)
        except Exception as e:
            print(f"Unexpected error: {str(e)}\n{sys.exc_info()}")

if __name__ == '__main__':
    main()