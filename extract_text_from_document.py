import os
import re
import csv

def extract_pattern(pattern, directory):
    extracted_data = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        extracted_data.append((filename, match))
    
    return extracted_data

if __name__ == "__main__":
    input_path = "C:\\python\\autoindex\\txt_output"
    output_path = "C:\\python\\autoindex\\extract"

    os.makedirs(output_path, exist_ok=True)

    # Define patterns
    social_sec_pattern = r"\b\d{3}-\d{2}-\d{4}\b|^\d{9}$"
    credit_card_pattern = r"(\d{4}[-\s]*?){4}"
    name_pattern = r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b"
    dl_number_pattern = r"\b(?:[A-Z][0-9]{3}|[A-K][0-9]{7})\b"  # A simplified version of a possible DL number pattern. Adjust as needed

    # Extract information
    social_sec_numbers = extract_pattern(social_sec_pattern, input_path)
    credit_card_numbers = extract_pattern(credit_card_pattern, input_path)
    specific_names = extract_pattern(name_pattern, input_path)
    dl_numbers = extract_pattern(dl_number_pattern, input_path)

    # Save extracted data in CSV format
    with open(os.path.join(output_path, "social_sec_numbers.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Social_Security_Number"])
        writer.writerows(social_sec_numbers)

    with open(os.path.join(output_path, "credit_card_numbers.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Credit_Card_Number"])
        writer.writerows(credit_card_numbers)

    with open(os.path.join(output_path, "specific_names.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Name"])
        writer.writerows(specific_names)

    with open(os.path.join(output_path, "dl_numbers.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Driver_License_Number"])
        writer.writerows(dl_numbers)

    # Show console output
    print("Extracted Social Security Numbers:")
    for entry in social_sec_numbers:
        print(f"{entry[0]}: {entry[1]}")

    print("\nExtracted Credit Card Numbers:")
    for entry in credit_card_numbers:
        print(f"{entry[0]}: {entry[1]}")

    print("\nExtracted Specific Names:")
    for entry in specific_names:
        print(f"{entry[0]}: {entry[1]}")

    print("\nExtracted Driver License Numbers:")
    for entry in dl_numbers:
        print(f"{entry[0]}: {entry[1]}")