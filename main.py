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

    # Call categorize_files from similarity_clustering
    similarity_clustering.categorize_files(directory_path, processed_texts, file_names, original_texts)

    subprocess.call(['python', 'audioExtractText.py'])  
    print("Transcribing Audio Files to Text...")
    subprocess.call(['python', 'dl_email.py'])
    print("Extracting emails to Text")
    subprocess.call(['python', 'autoextract.py'])  
    print("Extracting Named Entities...")
    subprocess.call(['python', 'autokvextract.py'])  
    print("Extracting Key / Value Pairs...")
    subprocess.call(['python', 'autoner.py'])  
    print("Performing Sentiment Analysis...")
    subprocess.call(['python', 'autosentiment.py'])
    print("Performing Summarization...")
    subprocess.call(['python', 'autosummarize.py'])

if __name__ == "__main__":
    main()