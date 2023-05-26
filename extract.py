import re
import sys
import os
from pathlib import Path

input_folder = "c:\\python\\autoindex\\txt_output"
output_folder = "c:\\python\\autoindex\\kvextract"

def extract_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    data = {}

    owner_pattern = r'(?i)\*OWNER OF MARK\s*([\w\s,]+)'
    email_pattern = r'(?i)\*EMAIL ADDRESS\s*([\w\.-]+@[\w\.-]+)'
    mark_pattern = r'(?i)\*MARK\s*([\w\s]+)'
    phone_pattern = r'(?i)PHONE\s*([\d-]+)'
    mailing_address_pattern = r'(?i)\*MAILING ADDRESS\s*([\w\s\d,\.]+)'
    city_pattern = r'(?i)\*CITY\s*([\w\s]+)'
    state_pattern = r'(?i)\*STATE\s*.+\n([\w\s]+)'
    signature_pattern = r'(?i)\* SIGNATURE\s*([\w\s\.-]+)'
    signatory_name_pattern = r'(?i)\* SIGNATORY\'S NAME\s*([\w\s\.-]+)'
    signatory_position_pattern = r'(?i)\* SIGNATORY\'S POSITION\s*([\w\s\.-]+)'
    date_signed_pattern = r'(?i)\* DATE SIGNED\s*([\w\s\d,/]+)'

    try:
        data['owner'] = re.findall(owner_pattern, content)[0].strip()
        data['email'] = re.findall(email_pattern, content)[0].strip()
        data['mark'] = re.findall(mark_pattern, content)[0].strip()
        data['phone'] = re.findall(phone_pattern, content)[0].strip()
        data['mailing_address'] = re.findall(mailing_address_pattern, content)[0].strip()
        data['city'] = re.findall(city_pattern, content)[0].strip()
        data['state'] = re.findall(state_pattern, content)[0].strip()
        data['signature'] = re.findall(signature_pattern, content)[0].strip()
        data['signatory_name'] = re.findall(signatory_name_pattern, content)[0].strip()
        data['signatory_position'] = re.findall(signatory_position_pattern, content)[0].strip()
        data['date_signed'] = re.findall(date_signed_pattern, content)[0].strip()
    except IndexError:
        print(f"Error: Couldn't extract data from {file_path}. Skipping file.")
        return None

    return data

def process_files(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not output_path.exists():
        os.makedirs(output_path)

    for file in input_path.glob("*.txt"):
        extracted_data = extract_data(file)
        if extracted_data:
            output_file = output_path / file.name
            with open(output_file, "w") as f:
                for key, value in extracted_data.items():
                    f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    process_files(input_folder, output_folder)