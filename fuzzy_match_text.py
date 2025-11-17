import pandas as pd
from thefuzz import process, fuzz
import os
import chardet
import re

def main():
    # Get paths from environment or use defaults
    txt_output_dir = os.getenv("TXT_OUTPUT_DIR", "txt_output")
    accounts_csv = os.getenv("ACCOUNTS_CSV", "accounts/accounts.csv")

    if not os.path.exists(txt_output_dir):
        print(f"Error: Directory '{txt_output_dir}' does not exist.")
        return

    if not os.path.exists(accounts_csv):
        print(f"Error: Accounts CSV file '{accounts_csv}' does not exist.")
        return

    # Read the plain text documents from the folder
    docs = []
    for file in os.listdir(txt_output_dir):
        if file.endswith(".txt"):
            with open(os.path.join(txt_output_dir, file), "rb") as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"]
                doc_text = raw_data.decode(encoding)

                if "invoice" in doc_text.lower():
                    doc_type = "Invoice"
                elif "receipt" in doc_text.lower():
                    doc_type = "Receipt"
                elif "statement" in doc_text.lower():
                    doc_type = "Statement"
                else:
                    doc_type = "Unknown"

                date_pattern = r"\d{2}/\d{2}/\d{4}"
                date_match = re.search(date_pattern, doc_text)
                if date_match:
                    doc_date = date_match.group()
                else:
                    doc_date = "Unknown"

                email_pattern = r"\w+@\w+\.\w+"
                email_match = re.search(email_pattern, doc_text)
                if email_match:
                    doc_sender = email_match.group()
                else:
                    doc_sender = "Unknown"

                docs.append({"doc_name": file[:-4], "doc_text": doc_text, "doc_type": doc_type, "doc_date": doc_date, "doc_sender": doc_sender})

    docs = pd.DataFrame(docs)

    # Read the table of account identifiers
    accounts = pd.read_csv(accounts_csv)

    # Define a list of folders that contain the extracted named entities
    ner_extraction_dir = os.getenv("NER_EXTRACTION_DIR", "NER_Extraction")
    ner_dir = os.getenv("NER_DIR", "NER")
    kvextract_dir = os.getenv("KVEXTRACT_DIR", "kvextract")
    rake_extraction_dir = os.getenv("RAKE_EXTRACTION_DIR", "Rake_Extraction")
    extract_dir = os.getenv("EXTRACT_DIR", "extract")

    ner_folders = [ner_extraction_dir, ner_dir, kvextract_dir, rake_extraction_dir, extract_dir]

    # Define a function to match a document with an account using fuzzy string matching
    def match_doc_with_account(doc):
        entities = []
        for folder in ner_folders:
            file_path = os.path.join(folder, doc["doc_name"] + ".txt")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    entities.extend(f.read().splitlines())

        matches = []

        for entity in entities:
            name_matches = process.extract(entity, choices=accounts["name"], scorer=fuzz.token_sort_ratio, limit=None)
            for name_match in name_matches:
                if name_match[1] >= 10:
                    print(f"Name match for '{doc['doc_name']}': Entity = '{entity}', Match = '{name_match[0]}', Score = {name_match[1]}")
                    matches.append(name_match)

            address_matches = process.extract(entity, choices=accounts["address"], scorer=fuzz.token_sort_ratio, limit=None)
            for address_match in address_matches:
                if address_match[1] >= 10:
                    print(f"Address match for '{doc['doc_name']}': Entity = '{entity}', Match = '{address_match[0]}', Score = {address_match[1]}")
                    matches.append(address_match)

        if matches:
            scores = pd.DataFrame(matches, columns=["account_number", "score", "match"])
            best_match = max(matches, key=lambda x: x[1], default=(None, None))

            matching_account = accounts.loc[accounts["name"] == best_match[0], "category"]
            category = matching_account.values[0] if not matching_account.empty else None

            confidence_score = best_match[1]

        else:
            best_match = None
            category = None
            confidence_score = None

        return {"account_number": best_match, "category": category, "confidence_score": confidence_score}

    new_data = docs.apply(match_doc_with_account, axis=1, result_type='expand')

    # Rename columns in new_data
    new_data.columns = ["account_number", "category", "confidence_score"]

    # Concatenate docs and new_data DataFrames
    results = pd.concat([docs, new_data], axis=1)

    # Save the results dataframe to a CSV file with a semicolon separator and only doc_name, account_number, category, and confidence_score columns
    results_csv = os.getenv("RESULTS_CSV", "results.csv")
    results.to_csv(results_csv, sep=";", index=False, columns=["doc_name", "account_number", "category", "confidence_score"])

    # Read the CSV file with a semicolon separator
    results = pd.read_csv(results_csv, sep=";")

    # Print the results
    print(results)

if __name__ == "__main__":
    main()