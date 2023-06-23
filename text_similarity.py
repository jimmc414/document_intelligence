import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set folder path
folder_path = "C:/python/autoindex/txt_output"

# Prompt user for filename
input_file = input("Enter the name of the text file: ")
input_file_path = os.path.join(folder_path, input_file)

# Read input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = file.read()

# Read all text files in the folder
corpus = []
file_names = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file_path.endswith(".txt") and file_path != input_file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_text = file.read()
        file_names.append(os.path.basename(file_path))
        corpus.append(file_text)

# Add input file to the corpus as the first item
corpus.insert(0, input_text)

# Vectorize texts using Tf-idf
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Calculate cosine similarity between input file and each text document
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Create a list of tuples (filename, similarity score)
similarity_report = list(zip(file_names, similarity_score[0]))

# Set similarity threshold
similarity_threshold = float(input("Enter similarity threshold (e.g. 0.5): "))

# Get documents above the similarity threshold
above_threshold = [item for item in similarity_report if item[1] >= similarity_threshold]

# Custom function to print report
def print_report(report, title):
    report_output = title + "\n"
    for item in report:
        report_output += f"{item[0]} - Similarity Score: {item[1]}\n"
    return report_output

# Print and save the filtered similarity report
with open("output.txt", "w", encoding='utf-8') as file:
    above_threshold_output = print_report(above_threshold, "Documents Above Threshold:")
    print(above_threshold_output)
    file.write(above_threshold_output)