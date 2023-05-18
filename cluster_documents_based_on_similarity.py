import sys
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import manage_files
import extract_information_from_text

# You can modify the `specific_phrases` list to include different phrases or add more phrases that you want to search for in the text files. Changing these phrases will affect the extraction of information and subsequent categorization of the text files based on the new specific_phrases.

# The `most_freq_phrases` list is dependent on the specific_phrases and the clustering process. If you change the specific_phrases, the most frequent phrases for each cluster may change accordingly.

# You can modify the condition inside the last `for` loop (`if most_freq_phrases[category].lower() == "request to file foreign judgment":`) to process the text files based on other most_freq_phrases, e.g., if you want to process files with the most frequent phrase being "resume" instead.

# By adding, removing, or changing phrases in `specific_phrases`, you will directly affect the extraction and categorization process, which in turn will affect the output of the program.


# The `TfidfVectorizer` plays a crucial role in this program, as it takes the `extracted_texts` containing the specific phrases and converts them into a format suitable for the clustering process. By considering the frequency of the specific phrases within and between the text files, the TF-IDF representation captures the importance of each phrase for each file, which helps the KMeans algorithm to group similar files together based on these specific phrases.


def determine_best_similarity_measure():
    return "cosine"

def optimal_clusters(x):
    return 3

def extract_specific_values(texts, specific_phrases):
    extracted_texts = []
    for text in texts:
        extracted_text = []
        for phrase in specific_phrases:
            if phrase.lower() in text.lower():
                extracted_text.append(phrase.lower())
        extracted_texts.append(" ".join(extracted_text))
    return extracted_texts

def get_most_frequent_phrase(cluster_labels, texts, specific_phrases):
    phrase_count_list = [Counter() for _ in range(max(cluster_labels) + 1)]
    for label, text in zip(cluster_labels, texts):
        for phrase in specific_phrases:
            if phrase.lower() in text.lower():
                phrase_count_list[label][phrase.lower()] += 1
    most_frequent_phrases = []
    for phrase_count in phrase_count_list:
        most_freq_phrase = phrase_count.most_common(1)
        if most_freq_phrase:
            most_frequent_phrases.append(most_freq_phrase[0][0])
        else:
            most_frequent_phrases.append("")
    return most_frequent_phrases

def cluster_analysis(texts, specific_phrases):
    cluster_algorithm = KMeans
    similarity_measure = determine_best_similarity_measure()
    n_clusters = optimal_clusters(texts)
    
    extracted_texts = extract_specific_values(texts, specific_phrases)
    vectorizer = TfidfVectorizer(analyzer='word', lowercase=True, use_idf=True, min_df=2, max_df=50)
    non_empty_texts = [text for text in extracted_texts if text.strip()]
    if not non_empty_texts:
        return None, None
    X = vectorizer.fit_transform(non_empty_texts)
    
    if not vectorizer.vocabulary_:
        return None, None
    
    cluster_model = cluster_algorithm(n_clusters=n_clusters)
    cluster_labels = cluster_model.fit_predict(X)
    
    most_frequent_phrases = get_most_frequent_phrase(cluster_labels, texts, specific_phrases)
    
    return cluster_labels, most_frequent_phrases

def categorize_files(input_directory, output_directory):
    # Read all the text files in the input_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_names = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
    original_texts = [manage_files.read_file(os.path.join(input_directory, f)) for f in file_names]

    # Specify the phrases you want to search for in the text files (can be modified)
    specific_phrases = [
        "REQUEST TO FILE FOREIGN JUDGMENT",
        "NOTICE OF SATISFACTION OF LIEN",
        "REQUEST FOR SUMMONS",
        "Resume",
    ]
    cluster_labels, most_freq_phrases = cluster_analysis(original_texts, specific_phrases)

    if cluster_labels is None and most_freq_phrases is None:
        print("Skipping further processing as no specific phrases found.")
        return

    results = {}
    for label, file_name in zip(cluster_labels, file_names):
        if label not in results:
            results[label] = []
        results[label].append(file_name)

    # Print categorized txt files and most frequent phrases
    for category, files in results.items():
        print(f"Category {category} (Most frequent phrase: '{most_freq_phrases[category]}'):")
        for file_name in files:
            print(f" {file_name}")
            # The condition for processing and outputting files can be modified based on most_freq_phrases
            if most_freq_phrases[category].lower() == "request to file foreign judgment":
                file_path = os.path.join(input_directory, file_name)
                content = manage_files.read_file(file_path)
                output_file_path = os.path.join(output_directory, file_name)
                manage_files.write_file(output_file_path, content)

                extracted_values = extract_information_from_text.extract_requested_info(content)
                for key, value in extracted_values.items():
                    print(f"  {key}: {value}")

input_directory = 'C:\\python\\autoindex\\txt_output'
output_directory = 'C:\\python\\autoindex\\classification'
categorize_files(input_directory, output_directory)

