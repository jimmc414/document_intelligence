import spacy
import re

def extract_case_number(text):
    case_number = re.search(r'(case no\.|case number)[:\s]*(\w+)', text, re.IGNORECASE)
    if case_number:
        return case_number.group(2)
    return ""

def extract_plaintiff(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""

def extract_address(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            return ent.text
    return ""

def extract_requested_info(text):
    case_number = extract_case_number(text)
    plaintiff_name = extract_plaintiff(text)
    address = extract_address(text)
    
    return {
        "Case Number": case_number,
        "Plaintiff Name": plaintiff_name,
        "Address": address
    }