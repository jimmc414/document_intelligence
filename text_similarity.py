import os
import gensim.downloader as api
import numpy as np
from scipy import spatial
from nltk import word_tokenize
from nltk.corpus import stopwords
import configparser
import pickle
import shutil

# Read settings from ini file
config = configparser.ConfigParser()
config.read('settings.ini')
folder_path = config.get('paths', 'txt_documents')

# Prompt user for filename
input_file = input("Enter the name of the text file: ")
input_file_path = os.path.join(folder_path, input_file)

# Set similarity threshold
similarity_threshold = float(input("Enter similarity threshold (e.g. 0.5): "))

# Ask user if they want to copy similar files to a subfolder
move_files = input("Do you want to copy similar files to a subfolder? (yes/no): ")

stop_words = stopwords.words('english')

def preprocess(text):
    return [word for word in word_tokenize(text.lower()) if word not in stop_words and word.isalnum()]

def get_feature_vec(words, model):
    feature_vec = np.zeros((model.vector_size,), dtype="float32")
    num_words = 0.
    index_to_key_set = set(model.key_to_index)
    for word in words:
        if word in index_to_key_set:
            num_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if num_words > 0:
        feature_vec = np.divide(feature_vec, num_words)
    return feature_vec

def compute_similarity(vec1, vec2):
    if np.any(vec1) and np.any(vec2):  # check vectors are not all zeros
        return 1 - spatial.distance.cosine(vec1, vec2)
    else:
        return 0  # return zero similarity if any of vectors is all zeros

# Check if Word2Vec model is already downloaded
model_name = 'word2vec-google-news-300'
model_path = f"{model_name}.pkl"
if os.path.exists(model_path):
    print("Using previously downloaded Word2Vec model from", model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    print("Downloading Word2Vec model to", model_path)
    model = api.load(model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# Read input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    input_text = preprocess(file.read())

# Vectorize input text
input_vec = get_feature_vec(input_text, model)

# Read all text files in the folder
corpus = []
file_names = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file_path.endswith(".txt") and file_path != input_file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_text = preprocess(file.read())
        file_names.append(os.path.basename(file_path))
        corpus.append(get_feature_vec(file_text, model))

# Calculate cosine similarity between input file and each text document
similarity_scores = [compute_similarity(input_vec, text_vec) for text_vec in corpus]

# Create a list of tuples (filename, cosine similarity)
similarity_report = list(zip(file_names, similarity_scores))

# Get documents above the similarity threshold
above_threshold = [item for item in similarity_report if item[1] >= similarity_threshold]

# Custom function to print report
def print_report(report, title):
    report_output = title + "\n"
    for item in report:
        report_output += f"{item[0]} - Similarity Score: {item[1]}\n"
    return report_output

# Print and write the filtered similarity report to a file
with open("output.txt", "w", encoding='utf-8') as file:
    above_threshold_output = print_report(above_threshold, "Documents Above Threshold:")
    print(above_threshold_output)
    file.write(above_threshold_output)

# If user wants to move similar files to a subfolder
if move_files.lower() == 'yes':
    subfolder_path = os.path.join(folder_path, os.path.splitext(input_file)[0])
    os.makedirs(subfolder_path, exist_ok=True)
    for file_name, _ in above_threshold:
        shutil.copy(os.path.join(folder_path, file_name), subfolder_path)