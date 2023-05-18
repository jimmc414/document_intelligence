import os
import manage_files
import preprocess_text
import extract_features_from_text
import extract_information_from_text
import sys
import subprocess
import json

directory_path = "c:/python/autoindex/txt_output"

def main():
    # running autoextractpdf2text.py
    subprocess.call(['python', 'extract_text_from_pdf.py'])
    
    # running autoocr_parallel.py
    subprocess.call(['python', 'optical_character_recognition.py'])

    txt_files = manage_files.get_txt_files(directory_path)
    processed_texts = []
    file_names = []
    original_texts = []
    for file_path in txt_files:
        content = manage_files.read_file(file_path)
        original_texts.append(content)
        tokens = preprocess_text.preprocess_text(content)
        vector = extract_features_from_text.vectorize_text(tokens)
        if vector is not None:
            processed_texts.append(vector.tolist()) # Convert numpy array to list
            file_names.append(os.path.basename(file_path))

    # Call categorize_files from cluster_documents_based_on_similarity
    # Save the arguments to a temporary file
    args = [directory_path, processed_texts, file_names, original_texts]
    temp_file = "temp_args.json"
    with open(temp_file, "w") as f:
        json.dump(args, f)
    
    # Pass the file name to the subprocess call
    subprocess.call(['python', 'cluster_documents_based_on_similarity.py', temp_file])

    subprocess.call(['python', 'extract_text_from_audio.py'])  
    print("Transcribing Audio Files to Text...")
    subprocess.call(['python', 'download_email.py'])
    print("Extracting emails to Text")
    subprocess.call(['python', 'extract_text_from_document.py'])  
    print("Extracting Named Entities...")
    subprocess.call(['python', 'extract_key_value_pairs.py'])  
    print("Extracting Key / Value Pairs...")
    subprocess.call(['python', 'extract_named_entities.py'])  
    print("Performing Sentiment Analysis...")
    subprocess.call(['python', 'sentiment_analysis.py'])
    print("Performing Summarization...")
    subprocess.call(['python', 'summarize_text.py'])
    print("Performing Classification...")
    subprocess.call(['python', 'cluster_documents.py'])
    

if __name__ == "__main__":
    main()