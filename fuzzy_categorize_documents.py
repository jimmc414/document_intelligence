import os
import shutil
import configparser
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

config = configparser.ConfigParser()
config.read("settings.ini")

TXT_DOCUMENTS = config.get("paths", "txt_documents")
SIMILARITY_THRESHOLD = config.getfloat("similarity", "similarity_threshold")
CATEGORY_DEFINITIONS = dict(config.items("categories"))


def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def move_file_to_subfolder(file_path, category):
    category_folder = os.path.join(os.path.dirname(file_path), category)
    os.makedirs(category_folder, exist_ok=True)

    new_file_path = os.path.join(category_folder, f"{category}_{os.path.basename(file_path)}")
    shutil.move(file_path, new_file_path)


def find_most_similar_category(text, categories, threshold):
    # Change the scorer here to try different matching algorithms.
    
    # Option 1: fuzz.token_set_ratio
    # most_similar_category, highest_similarity = process.extractOne(text, categories.values(), scorer=fuzz.token_set_ratio)

    # Option 2: fuzz.token_sort_ratio
    most_similar_category, highest_similarity = process.extractOne(text, categories.values(), scorer=fuzz.token_sort_ratio)

    similarities = process.extract(text, categories.values(), scorer=fuzz.token_sort_ratio)
    for cat_def, score in similarities:
        cat_name = [k for k, v in categories.items() if v == cat_def][0]
        print(f"Similarity between document and category '{cat_name}': {score}")

    if highest_similarity >= threshold:
        return [k for k, v in categories.items() if v == most_similar_category][0]
    else:
        return None


if __name__ == "__main__":
    with open("exception.txt", "w", encoding="utf-8") as exception_file:
        for filename in os.listdir(TXT_DOCUMENTS):
            if filename.endswith(".txt"):
                file_path = os.path.join(TXT_DOCUMENTS, filename)
                txt_text = extract_text_from_txt(file_path)

                most_similar_category = find_most_similar_category(txt_text, CATEGORY_DEFINITIONS, SIMILARITY_THRESHOLD)

                print(
                    f"Processing {filename}: Most similar category: {most_similar_category}"
                )

                if most_similar_category is not None:
                    move_file_to_subfolder(file_path, most_similar_category)
                    print(f"{filename} categorized as {most_similar_category}")
                else:
                    exception_file.write(f"{filename}\n")
                    print(f"{filename} NOT categorized (below threshold)")

    print("Categorization process completed.")