import spacy
from spacy.pipeline import EntityRuler
import os
import re
from spacy.matcher import Matcher

# Function to extract proper names
def extract_proper_names(doc):
    proper_names = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    return proper_names

# Function to extract case numbers and account numbers from the text
def extract_identifiers(text, case_pattern, account_pattern):
    case_numbers = re.findall(case_pattern, text)
    account_numbers = re.findall(account_pattern, text)

    return case_numbers, account_numbers

# Function to extract legal terms
def extract_legal_terms(doc, legal_terms):
    matcher = Matcher(nlp.vocab)
    patterns = [[{"LOWER": term.lower()}] for term in legal_terms]
    matcher.add("LegalTerms", patterns)

    matches = matcher(doc)
    terms = [doc[start:end].text for match_id, start, end in matches]

    return terms

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Custom pattern for cases like "Fname M Lname"
custom_name_pattern = [
    {
        "label": "PERSON",
        "pattern": [
            {"POS": "PROPN"},
            {"ORTH": ".", "OP": "?"},
            {"POS": "PROPN"},
        ]
    }
]

# Add EntityRuler to the pipeline
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(custom_name_pattern)

# Set the directory paths
input_dir = "C:\\python\\autoindex\\txt_output"
output_dir = "C:\\python\\autoindex\\NER"

# Define patterns for case numbers and account numbers
case_pattern = r'\bCASE-\d{4}-\d{3}\b'  # Sample pattern for case numbers
account_pattern = r'\bACCN\d{5}\b'  # Sample pattern for account numbers

# Define a list of legal terms
legal_terms = ["Plaintiff", "Defendant", "Claim", "Relief", "Negligence"]

# Iterate through txt files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_dir, filename)

        # Load the text from the txt file
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Process the text using spaCy
        doc = nlp(text)

        # Extract only proper names
        proper_names = extract_proper_names(doc)

        # Extract case numbers and account numbers
        case_numbers, account_numbers = extract_identifiers(text, case_pattern, account_pattern)

        # Extract legal terms
        legal_term_matches = extract_legal_terms(doc, legal_terms)

        # Check if any identifiers were found for the file
        if proper_names or case_numbers or account_numbers or legal_term_matches:
            # Print identifiers for each document
            print(f"Identifiers in {filename}:")

            if proper_names:
                print("Proper Names:")
                for name in proper_names:
                    print(name.text)

            if case_numbers:
                print("Case Numbers:")
                for number in case_numbers:
                    print(number)

            if account_numbers:
                print("Account Numbers:")
                for number in account_numbers:
                    print(number)

            if legal_term_matches:
                print("Legal Terms:")
                for term in legal_term_matches:
                    print(term)

            output_filename = "NER_" + filename
            output_filepath = os.path.join(output_dir, output_filename)

            # Write the identifiers to the output file
            with open(output_filepath, "w", encoding="utf-8") as output_file:
                if proper_names:
                    output_file.write("Proper Names:\n")
                    for name in proper_names:
                        output_file.write(name.text + "\n")

                if case_numbers:
                    output_file.write("\nCase Numbers:\n")
                    for number in case_numbers:
                        output_file.write(number + "\n")

                if account_numbers:
                    output_file.write("\nAccount Numbers:\n")
                    for number in account_numbers:
                        output_file.write(number + "\n")

                if legal_term_matches:
                    output_file.write("\nLegal Terms:\n")
                    for term in legal_term_matches:
                        output_file.write(term + "\n")