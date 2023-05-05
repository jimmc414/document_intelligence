import os
import file_utils
import text_preprocessing
import feature_extraction
import similarity_clustering
import info_extraction
import sys
import subprocess

directory_path = "c:/python/autoindex/txt_output"

def main():
    # running autoextractpdf2text.py
    subprocess.call(['python', 'autoextractpdf2text.py'])
    
    # running autoocr_parallel.py
    subprocess.call(['python', 'autoocr_parallel.py'])

    txt_files = file_utils.get_txt_files(directory_path)
    processed_texts = []
    file_names = []
    original_texts = []
    for file_path in txt_files:
        content = file_utils.read_file(file_path)
        original_texts.append(content)
        tokens = text_preprocessing.preprocess_text(content)
        vector = feature_extraction.vectorize_text(tokens)
        if vector is not None:
            processed_texts.append(vector)
            file_names.append(os.path.basename(file_path))

    specific_phrases = ["REQUEST TO FILE FOREIGN JUDGMENT", "NOTICE OF SATISFACTION OF LIEN", "REQUEST FOR SUMMONS", "Resume"]
    cluster_labels, most_freq_phrases = similarity_clustering.cluster_analysis(original_texts, specific_phrases)

    if cluster_labels is None and most_freq_phrases is None:
        print("Skipping further processing as no specific phrases found.")
    else:
        results = {}
        for label, file_name in zip(cluster_labels, file_names):
            if label not in results:
                results[label] = []
            results[label].append(file_name)

        print("Categorized txt files based on similarity:")
        for category, files in results.items():
            print(f"Category {category} (Most frequent phrase: '{most_freq_phrases[category]}'):")
            for file_name in files:
                print(f" {file_name}")
                if most_freq_phrases[category].lower() == "request to file foreign judgment":
                    file_path = os.path.join(directory_path, file_name)
                    content = file_utils.read_file(file_path)
                    
                    extracted_values = info_extraction.extract_requested_info(content)
                    for key, value in extracted_values.items():
                        print(f"  {key}: {value}")
                        
    subprocess.call(['python', 'autoner.py'])  # Added this line to call autoner.py
    print("Performing Sentiment Analysis...")
    subprocess.call(['python', 'autosentiment.py'])
    print("Performing Summarization...")
    subprocess.call(['python', 'autosummarize.py'])
    
if __name__ == "__main__":
    main()