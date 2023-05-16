import os
import glob

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def get_txt_files(directory_path):
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    return txt_files