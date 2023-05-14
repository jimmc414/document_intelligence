import os
from rake_nltk import Rake

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\Rake_Extraction"

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords

def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)

            with open(file_path, "r", encoding="utf-8") as input_file:
                text = input_file.read()

            results = extract_keywords(text)

            output_file_name = f"Rake_Extraction_{file}"
            output_file_path = os.path.join(output_dir, output_file_name)

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for keyword in results:
                    output_file.write(f"{keyword}\n")

os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)