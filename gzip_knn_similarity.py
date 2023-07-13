import os
import numpy as np
import gzip
from nltk import word_tokenize
from nltk.corpus import stopwords
import configparser

# Read settings from ini file
config = configparser.ConfigParser()
config.read('settings.ini')
folder_path = config.get('paths', 'txt_documents')

# Prompt user for filename
input_file = input("Enter the name of the text file: ")
input_file_path = os.path.join(folder_path, input_file)

# Set distance threshold
distance_threshold = float(input("Enter distance threshold (e.g. 0.5): "))

stop_words = stopwords.words('english')

def preprocess(text):
    return [word for word in word_tokenize(text.lower()) if word not in stop_words and word.isalnum()]

def compute_gzip_distance(x1, x2):
    x1_compressed = gzip.compress(x1.encode())
    x2_compressed = gzip.compress(x2.encode())
    concat_compressed = gzip.compress(' '.join([x1, x2]).encode())
    return (len(concat_compressed) - min(len(x1_compressed), len(x2_compressed))) / max(len(x1_compressed), len(x2_compressed))

# Read input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = ' '.join(preprocess(file.read()))

# Read all text files in the folder
corpus = []
file_names = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file_path.endswith(".txt") and file_path != input_file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_text = ' '.join(preprocess(file.read()))
        file_names.append(os.path.basename(file_path))
        corpus.append(file_text)

# Calculate gzip distance between input file and each text document
distance_scores = [compute_gzip_distance(input_text, text) for text in corpus]

# Create a list of tuples (filename, gzip distance)
distance_report = list(zip(file_names, distance_scores))

# Get documents below the distance threshold
below_threshold = [item for item in distance_report if item[1] <= distance_threshold]

# Custom function to print report
def print_report(report, title):
    report_output = title + "\\n"
    for item in report:
        report_output += f"{item[0]} - Distance Score: {item[1]}\\n"
    return report_output

# Print and write the filtered distance report to a file
with open("output.txt", "w", encoding='utf-8') as file:
    below_threshold_output = print_report(below_threshold, "Documents Below Threshold:")
    print(below_threshold_output)
    file.write(below_threshold_output)
