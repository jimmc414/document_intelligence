import os
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def ner_extraction(text):
    doc = nlp(text)

    # Extract named entities
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract PERSON names and account number using regex
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    account_numbers = re.findall(r'\d+', text)

    return named_entities, persons, account_numbers

def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)
            with open(file_path, "r", encoding="utf-8") as input_file:
                text = input_file.read()

            named_entities, persons, account_numbers = ner_extraction(text)

            output_file_path = os.path.join(output_dir, f"NER_Extraction_{file}")

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write("Named entities:\n")
                for entity, label in named_entities:
                    output_file.write(f"{entity}: {label}\n")
                output_file.write("\nPersons:\n")
                for person in persons:
                    output_file.write(f"{person}\n")
                output_file.write("\nAccount numbers:\n")
                for acc_num in account_numbers:
                    output_file.write(f"{acc_num}\n")

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\NER_Extraction"
os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)