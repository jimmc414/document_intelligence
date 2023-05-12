---

# AutoIndex: PDF Text Analysis and Processing

AutoIndex is a Document Intelligence Proof of Concept designed to automatically process PDF files. It extracts relevant information, analyzes sentiment, clusters, and categorizes documents based on similarity, generates summaries, and extracts text.

This information will be used to automatically assign the correct account number in an external system.

This project aims to automatically digest unstructured documents, phone calls, or texts and generate metadata that indexes the correct account in a remote system. The objective is for this process to occur end-to-end, from unknown document to correctly filed, without human intervention.

The project consists of several Python scripts that accomplish different tasks, including OCR processing, named entity recognition, similarity clustering, sentiment analysis, and text summarization.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- PyMuPDF: A Python library that enables you to access and manipulate PDF files, including text extraction, metadata retrieval, and layout analysis.
- pdf2image: A Python library that converts PDF files into images, allowing for easier processing and analysis of non-textual content in the files.
- PyTesseract: A Python wrapper for the Tesseract OCR engine, which enables Optical Character Recognition (OCR) capabilities for converting scanned images and PDFs into text.
- NLTK: The Natural Language Toolkit (NLTK) is a Python library for working with human language data, providing tools for text processing, tokenization, parsing, and classification.
- Spacy: A fast and modern Python library for natural language processing (NLP), offering pretrained models and extensive functionality for NLP tasks such as text tokenization, part-of-speech tagging, and named entity recognition.
- Gensim: A Python library specialized in topic modeling and document similarity analysis, which allows for efficient large-scale text processing using algorithms like Word2Vec, FastText, and Latent Semantic Analysis.
- TextBlob: A simple Python library for processing textual data, offering basic NLP features such as part-of-speech tagging, noun phrase extraction, sentiment analysis, and translation.
- Sumy: A Python library that provides functionality to summarize texts using various algorithms, extracting key information from long articles and documents.
- Scikit-learn: A popular Python library for machine learning and data science, offering tools for preprocessing data, training algorithms, and evaluating model performance.
- Transformers (Hugging Face): A Python library for state-of-the-art natural language processing, based on transformer model architectures like BERT and GPT, and offering pre-trained models for tasks like text generation, question-answering, and sentiment analysis.
- OpenAI API - Whisper: An API service provided by OpenAI that offers an automatic speech recognition (ASR) system, Whisper, for transcribing spoken language from audio data into written text with high accuracy.

Use pip to install the required packages in your Python environment:

```
pip install PyMuPDF pdf2image pytesseract nltk spacy gensim textblob sumy scikit-learn transformers openai os
```

Download the necessary language model for Spacy:

```
python -m spacy download en_core_web_sm
```

To install the poppler files for this project to work, follow these steps:

1. Download the latest poppler package from https://pypi.org/project/pdf2image/. This is the most up-to-date version for Windows users.
2. Move the extracted directory to the desired location on your system.
3. Add the `bin/` directory to your PATH environment variable. 
5. Alternatively, use `poppler_path = r"C:\path\to\poppler-xx\bin"` as an argument in the `convert_from_path` function in your `autoextractpdf2text.py` script. 

Follow these steps to install Tesseract and set up the necessary dependencies for the Python project:

1. Install Tesseract OCR: Download and install Tesseract OCR from the official website (https://github.com/UB-Mannheim/tesseract/wiki). 

2. After installation, add the Tesseract OCR executable to your system's PATH. On Windows, the default installation path is `C:\Program Files\Tesseract-OCR\tesseract.exe`. Add this path to your system's PATH variable or configure the pytesseract library to use the installed Tesseract directly.

3. Verify Tesseract installation: In the Python script `audioextracttext.py`, ensure the following line points to the correct Tesseract executable location:

```python
pytesseract.pytesseract.tesseract_cmd = r'c:\program files\tesseract-ocr\tesseract.exe'
```

4. Finally, create the necessary directories for the project, such as 'documents', 'txt_output', 'ner', 'sentiments', 'summarization', as specified in the `main.py` script.

### Directory Structure

Create the following directory structure for the AutoIndex project:

```
- autoindex
  - documents (Store PDF files here)
  - txt_output
  - ner
  - sentiments
  - summarization
  - audio
  - audio_txt
```

Place your PDF files in the "documents" folder.

## Running the Project

Run the `main.py` script to start processing the PDF files:

```
python main.py
```

The script will run sequentially in the following order:

1. `autoextractpdf2text.py`: Extracts text from searchable PDF files
2. `autoocr_parallel.py`: Performs OCR on non-searchable PDF files
3. `autoner.py`: Extracts proper names from the text files
4. `autosentiment.py`: Generates sentiment analysis for text files
5. `autosummarize.py`: Computes text summarization for text files
6. `audioExtractText.py`: Transcribes audio files to text at rate of $.006/min (rounded to nearest second). Requires OpenAI API key

### Output

After running the `main.py` script, you will find the following output in their respective directories:

- Extracted text from PDF files in the "txt_output" folder
- Named entity recognition results in the "ner" folder
- Sentiment analysis results in the "sentiments" folder
- Text summarization results in the "summarization" folder
- Transcribed audio text results in the "audio_txt" folder

Copyright (c) [2023] [James M. McMillan III]

---

**Program:** `audioextracttext.py`  
**Called From:** Main Script  
**Description:** This program transcribes audio files in a specified directory to text files using the OpenAI `whisper` ASR model.

---

**Program:** `autoextract.py`  
**Called From:** Main Script  
**Description:** This program extracts specific patterns (e.g., Social Security numbers, Credit Card numbers, names, driver license) from text files in a specified directory and saves them into separate CSV files.

---

**Program:** `autoextractpdf2text.py`  
**Called From:** Main Script  
**Description:** This program extracts text from searchable PDF files in a specified directory and saves them as text files in another directory.

---

**Program:** `autoocr_parallel.py`  
**Called From:** Main Script  
**Description:** This program converts non-searchable PDF files into searchable text using OCR (Optical Character Recognition) with pytesseract and multithreading for parallel processing.

---

**Program:** `autokvextract.py`  
**Called From:** Main Script  
**Description:** This program extracts key-value pairs from text files in a given directory and writes them to new text files.

---

**Program:** `autoner.py`  
**Called From:** Main Script  
**Description:** This program performs Named Entity Recognition (NER) using spaCy on text files in a specified directory and writes the results to text files.

---

**Program:** `autosentiment.py`  
**Called From:** Main Script  
**Description:** This program performs sentiment analysis on text files in a specified directory using TextBlob and writes the results to text files.

---

**Program:** `autosummarize.py`  
**Called From:** Main Script  
**Description:** This program generates summaries for text files in a specified directory using sumy library and writes the summaries to text files.

---

**Program:** `convert_audio.py`  
**Called From:** N/A, standalone script  
**Description:** This program converts audio files from one format (e.g., AMR) to another format (e.g., WAV) in a specified directory.

---

**Program:** `dl_email.py`  
**Called From:** Main Script  
**Description:** This program downloads emails from a Gmail account based on search criteria and saves them as text files in a specified directory.

---

**Program:** `feature_extraction.py`  
**Called From:** Main Script  
**Description:** This program vectorizes text using pre-trained word2vec models (in this case, Google's pre-trained Word2Vec model).

---

**Program:** `file_utils.py`  
**Called From:** Other scripts  
**Description:** This program contains utility functions like reading text files and getting all the TXT files from a given directory.

---

**Program:** `info_extraction.py`  
**Called From:** Main Script   
**Description:** This program extracts specific information (case number, plaintiff, and address) from text files using regex and Named Entity Recognition (NER).

---

**Program:** `main.py`  
**Called From:** N/A, main script to run the application  
**Description:** This script coordinates the execution of all other scripts in the correct order, such as text extraction, entity extraction, OCR, NER, and others.

---

**Program:** `similarity_clustering.py`  
**Called From:** Main Script  
**Description:** This program performs similarity-based clustering on text files using TF-IDF and K-means clustering, categorizes files based on similarity, and identifies the most frequent phrase in each cluster.

---

**Program:** `text_preprocessing.py`  
**Called From:** Main Script  
**Description:** This program preprocesses text by performing tokenization, lowercasing, punctuation removal, stopword removal, and lemmatization.

1. Set up the environment:
   a. Install Python, if not already installed: https://www.python.org/downloads/.
   b. Install an Integrated Development Environment (IDE) (optional, but recommended): Some popular options are VSCode, PyCharm, and Jupyter Notebook.

2. Install the necessary libraries:
   a. Install Natural Language Toolkit (nltk) for text preprocessing: `pip install nltk`
   b. Install scikit-learn for machine learning algorithms: `pip install scikit-learn`
   c. Install the Gensim library for working with pre-trained word embeddings: `pip install gensim`
   d. Optional: Install any other desired NLP libraries like spaCy, TextBlob, or BERT.

3. Define the file scanning and reading functions:
   a. Import the necessary modules: os, nltk, glob, and re.
   b. Write a function to read the contents of a file.
   c. Write a function to scan the target directory for txt files.

4. Preprocessing:
   a. Import nltk functions: word_tokenize, stopwords, and WordNetLemmatizer.
   b. Write a function to clean and preprocess the text data:
      - Tokenize the text
      - Convert text to lowercase
      - Remove punctuation and special characters
      - Remove stopwords
      - Lemmatize words
   c. Preprocess each text file's content and save in a list.

5. Feature extraction:
   a. Use Gensim's Word2Vec or other pretrained word embeddings like GloVe or fastText to represent words as vectors.
   b. Write a function to vectorize text data using the pre-trained word embeddings.
   c. Transform the preprocessed text data into feature vectors.

6. Determine the best similarity measure:
   a. Study similarity measures like: cosine, Jaccard, Euclidean, or Pearson.
   b. Choose the most appropriate similarity measure for this task.

7. Cluster analysis:
   a. Decide on the clustering algorithm that best suits the project requirements (e.g., K-Means, DBSCAN, Agglomerative Clustering, etc.).
   b. Write a function that automatically determines the optimal number of clusters.
   c. Implement the clustering algorithm, considering the optimal number of clusters.

8. Categorize the txt files based on similarity:
   a. Assign each text file to its corresponding cluster.
   b. Save the clustering results into a convenient format (dictionary, CSV, etc.).

9. Test the program
   a. Create a test directory.
   b. Add some txt files to the test directory.
   c. Run the program on this test directory.
   d. Evaluate the results and improve the model as needed.

10. Optional: Implement a user interface
    a. Create a GUI or a web interface for users to easily use the program.
	
	