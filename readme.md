
"A business is simply an idea to make other people's lives better."

—Sir Richard Branson

---

# Document Intelligence: Document Text Analysis and Artificial Intelligence Processing
### Description
Document Intelligence is a document intelligence proof of concept that is designed to automatically process and analyze PDF files, emails and call recordings. It extracts relevant information, analyzes sentiment, clusters and categorizes documents based on similarity, and generates summaries. The information is used to automatically assign the correct account number to an external system. The project aims to automatically digest unstructured documents, such as phone calls and texts, and generate metadata indexes for the correct account in a remote system. The objective is for this process to occur end-to-end, from an unknown document to being correctly filed, without human intervention.

The project consists of several Python scripts that accomplish different tasks, including OCR processing, named entity recognition, similarity clustering, sentiment analysis, text summarization, audio transcription, email extraction, key-value pair extraction, and classification.

### Installation / How to setup

You can use pip to install the required packages in your Python environment:

```
pip install textblob sys sklearn torch spacy csv gensim thefuzz threading transformers numpy chardet stanza glob google_auth_oauthlib pdf2image sumy time warnings flair vaderSentiment concurrent google fuzzywuzzy PyPDF4 re os openai base64 nltk pandas functools requests googleapiclient rake_nltk subprocess fitz pytesseract collections
```

You also need to download the necessary language model for spaCy:
```
python -m spacy download en_core_web_sm
```
Additionally, you need to install Poppler and Tesseract on your system and set the necessary dependencies for the Python project. To do this, follow these steps:

Download and install Poppler from https://pypi.org/project/pdf2image/. There is an up-to-date version for Windows users.

Move the extracted directory to a desired location on your system.

Add the bin/ directory to your path environment variable.
```
set PATH=%PATH%;C:\python\autoindex\poppler-0.68.0\bin
```
Alternatively, you can use the poppler_path = r"c:\path\to\poppler-xx\bin" argument in the convert_from_path function in the extract_text_from_pdf.py script.

Download and install Tesseract OCR from its official website (https://github.com/ub-mannheim/tesseract/wiki).

After installation, add the Tesseract OCR executable to your system’s path. For Windows, the default installation path is c:\program files\tesseract-ocr\tesseract.exe. Add this path to your system’s path variable or configure the PyTesseract library to use the installed Tesseract directly.
```
set PATH=%PATH%;c:\program files\tesseract-ocr\tesseract.exe
```
Verify your Tesseract installation: In the Python script.
```
import pytesseract
print(pytesseract.get_tesseract_version())
```
You should see something like tesseract 4.1.1 or higher.

To run the main script of the project, you can use the following command in your terminal:
```
python main.py
```

This will process all PDF documents in the 'documents' folder. The script will then perform the following tasks:

- Extract the text from the PDF file using PyMuPDF and PyTesseract.
- Analyze the text using spaCy, Gensim, TextBlob, Sumy, and Transformers.
- Generate a summary of the document and print it to the terminal.
- Extract the key-value pairs from the document and print them to the terminal.
- Cluster the document based on its similarity to other documents in a predefined corpus and print the cluster label to the terminal.
- Classify the document based on its sentiment and print the sentiment score to the terminal.
- Extract any email addresses from the document and print them to the terminal.
- Transcribe any audio files attached to the document using OpenAI API - Whisper and print the transcription to the terminal.
- All programs can be called as stand-alone processes.

Programs can also all run individually

## All Programs and Processes

**Programs Short Version**


| Program                 | Description                                                                                                               |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------|
| extract_text_from_audio.py| Extracts text from audio files. It uses the Google Cloud Speech API to transcribe the audio into text.                   |
| extract_text_from_document.py| Extracts information from documents. It can extract entities, dates, keywords, and summaries from documents.              |
| extract_text_from_pdf.py| Extracts text from PDF files. It uses the PyPDF2 library to convert PDF files to text.                                   |
| extract_key_value_pairs.py| Extracts key-value pairs from text. It uses the spacy library to identify key-value pairs in text.                        |
| extract_named_entities.py| Extracts named entities from text. It uses the spacy library to identify named entities in text.                           |
| optical_character_recognition.py| Extracts text from images using optical character recognition (OCR). It uses the Tesseract OCR library to extract text.   |
| sentiment_analysis.py| Analyzes the sentiment of text. It uses the VaderSentiment library to analyze the sentiment of text.                      |
| summarize_text.py| Summarizes text. It uses the sumy library to summarize text.                                                             |
| cluster_documents.py| Classifies text into categories using k-means clustering. It uses the scikit-learn library to perform k-means clustering. |
| convert_audio_format.py| Converts audio files from one format to another. It uses the ffmpeg library to convert audio files.                      |
| download_email.py| Downloads emails from a mail server. It uses the imaplib library to download emails from a mail server.                  |
| classify_documents.py| Classifies documents into categories. It uses the spacy library to extract features from documents and then uses a machine learning algorithm to classify the documents into categories. |
| compare_documents.py| Calculates the similarity between documents. It uses the gensim library to calculate the similarity between documents.    |
| extract_features_from_text.py| Extracts features from text. It uses the spacy library to extract features from text.                                      |
| manage_files.py | Provides utility functions for working with files. It can be used to read, write, and delete files.                      |
| sentiment_analysis_using_flair.py| Analyzes the sentiment of text using the FastText library.                                                                |
| compare_addresses.py| Compares addresses using the fuzzywuzzy library. It can be used to find similar addresses.                             |
| fuzzy_match_text.py| Performs fuzzy matching on text. It can be used to find similar text strings.                                             |
| sentiment_analysis_using_huggingface.py| Analyzes the sentiment of text using the Hugging Face Transformers library.                                               |
| extract_information_from_text.py| Extracts information from documents. It can extract entities, dates, keywords, and summaries from documents.              |
| main.py                  | This is the main program of the project. It provides a command line interface for the other programs.                     |
| extract_named_entities_using_spacy.py| Extracts named entities from text. It uses the spacy library to identify named entities in text.                           |
| extract_keywords_using_rake.py| Extracts keywords from text. It uses the Rake library to extract keywords from text.                                       |
| extract_relations_between_entities.py| Extracts relations between entities from text. It uses the spacy library to identify relations between entities in text.  |
| cluster_documents_based_on_similarity.py| Clusters documents based on their similarity. It uses the scikit-learn library to perform k-means clustering.             |
| preprocess_text.py| Preprocesses text. It can be used to clean, tokenize, and normalize text.                                                  |
| create_topic_model.py| Creates topic models from text. It uses the gensim library to create topic models from text.                               |
| sentiment_analysis_using_vader.py| Analyzes the sentiment of text using the VADER library.                                                                  |


**Programs Long Version**


extract_text_from_audio.py: A program that performs the following functions:
- Get an audio file as an input argument
- Transcribe the audio file using googleapiclient library
- Write the transcription to a text file with the same name as the audio file
- Called by: main.py or as standalone program

extract_text_from_document.py: A program that performs the following functions:
- Get a PDF file as an input argument
- Extract text from the PDF file using PyMuPDF library
- Write the extracted text to a text file with the same name as the PDF file
- Called by: autoExtractPDF2text.py

extract_text_from_pdf.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all PDF files in the input directory
- Call autoextract.py to extract text from each PDF file
- Called by: main.py or as standalone program

extract_key_value_pairs.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all text files in the input directory
- Preprocess each text file using text_preprocessing library
- Extract key-value pairs from each text file using info_extraction library
- Write the extracted key-value pairs to a CSV file
- Called by: main.py or as standalone program

extract_named_entities.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all text files in the input directory
- Preprocess each text file using text_preprocessing library
- Perform named entity recognition on each text file using spacy, flair, and transformers libraries
- Write the extracted entities to a CSV file
- Called by: main.py or as standalone program

optical_character_recognition.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all image files in the input directory
- Convert each image file to a PDF file using pdf2image library
- Perform OCR on each PDF file using pytesseract library
- Write the OCR text to a text file with the same name as the image file
- Use threading library to run multiple OCR processes in parallel
- Called by: main.py or as standalone program

sentiment_analysis.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all text files in the input directory
- Preprocess each text file using text_preprocessing library
- Perform sentiment analysis on each text file using vaderSentiment, flair, and transformers libraries
- Write the sentiment scores to a CSV file
- Called by: main.py or as standalone program

summarize_text.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all text files in the input directory
- Preprocess each text file using text_preprocessing library
- Generate summaries for each text file using sumy, transformers, and openai libraries
- Write the summaries to a CSV file
- Called by: main.py or as standalone program

cluster_documents.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all text files in the input directory
- Preprocess each text file using text_preprocessing library
- Embed each text file using torch, spacy, gensim, and transformers libraries
- Perform k-means clustering on the embeddings using sklearn library
- Compare the extracted keywords with a list of predefined categories using fuzzywuzzy and thefuzz libraries
- Assign the best matching category to each document
- Write the categories to a CSV file
- Called by: main.py or as standalone program

convert_audio_format.py: A program that performs the following functions:
- Get an audio file as an input argument
- Convert the audio file to a WAV format using subprocess library
- Write the converted audio file to a WAV file with the same name as the original file
- Called by: audioExtractText.py

download_email.py: A program that performs the following functions:
- Get an email address and a password as input arguments
- Connect to Gmail API using googleapiclient and google_auth_oauthlib libraries
- Download all attachments from unread emails in a specified folder
- Write the downloaded attachments to a specified directory
- Called by: main.py or as standalone program

classify_documents.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Embed the text file using torch and transformers libraries
- Load a pre-trained document classification model using torch and transformers libraries
- Predict the document class using the model
- Write the predicted class to a CSV file
- Called by: main.py or as standalone program

compare_documents.py: A program that performs the following functions:
- Get two text files as input arguments
- Preprocess the text files using text_preprocessing library
- Embed the text files using torch and transformers libraries
- Compute the cosine similarity between the embeddings using numpy library
- Write the similarity score to a CSV file
- Called by: main.py or as standalone program

extract_features_from_text.py: A module that defines some functions for extracting features from text data, such as TF-IDF vectors, word embeddings, and document embeddings.

manage_files.py: A module that defines some functions for working with files, such as reading, writing, deleting, moving, and renaming files.

sentiment_analysis_using_flair.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Load a pre-trained sentiment analysis model using flair library
- Predict the sentiment polarity and score using the model
- Write the predicted sentiment to a CSV file
- Called by: main.py or as standalone program

compare_addresses.py: A program that performs the following functions:
- Get two addresses as input arguments
- Compare the addresses using fuzzywuzzy library
- Write the comparison score to a CSV file
- Called by: main.py or as standalone program

fuzzy_match_text.py: A program that performs the following functions:
- Get a text file and a list of categories as input arguments
- Extract keywords from the text file using rake_nltk library
- Compare the keywords with the categories using fuzzywuzzy and thefuzz libraries
- Assign the best matching category to the text file
- Write the assigned category to a CSV file
- Called by: main.py or as standalone program

sentiment_analysis_using_huggingface.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Load a pre-trained sentiment analysis model using transformers library
- Predict the sentiment polarity and score using the model
- Write the predicted sentiment to a CSV file
- Called by: main.py or as standalone program

extract_information_from_text.py: A module that defines some functions for extracting information from text data, such as entities, dates, keywords, and key-value pairs.

main.py: The main program that calls other programs and performs the following functions:
- Get the input directory and output file name from the command-line arguments
- Check if the input directory exists and is not empty
- Call autoExtractPDF2text.py to extract text from PDF files
- Call autoOCR_parallel.py to perform OCR on image files
- Call autoNER.py to perform named entity recognition on text files
- Call categorize_kmeans.py to perform k-means clustering on text files
- Call autosummarize.py to generate summaries for text files
- Call autosentiment.py to perform sentiment analysis on text files
- Call autokvextract.py to extract key-value pairs from text files
- Write the results to a CSV file

extract_named_entities_using_spacy.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Perform named entity recognition on the text file using spacy, flair, and transformers libraries
- Write the extracted entities to a CSV file
- Called by: main.py or as standalone program

extract_keywords_using_rake.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Extract keywords from the text file using rake_nltk library
- Write the extracted keywords to a CSV file
- Called by: main.py or as standalone program

extract_relations_between_entities.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Perform relation extraction on the text file using spacy library
- Write the extracted relations to a CSV file
- Called by: main.py or as standalone program

cluster_documents_based_on_similarity.py: A program that performs the following functions:
- Get the input directory and output file name from the main program
- Iterate over all text files in the input directory
- Preprocess each text file using text_preprocessing library
- Embed each text file using torch, spacy, gensim, and transformers libraries
- Perform similarity clustering on the embeddings using similarity_clustering library
- Write the cluster labels to a CSV file
- Called by: main.py or as standalone program

preprocess_text.py: A module that defines some functions for preprocessing text data, such as cleaning, tokenizing, lemmatizing, and vectorizing.

create_topic_model.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Perform topic modeling on the text file using gensim library
- Write the extracted topics to a CSV file
- Called by: main.py or as standalone program

sentiment_analysis_using_vader.py: A program that performs the following functions:
- Get a text file as an input argument
- Preprocess the text file using text_preprocessing library
- Perform sentiment analysis on the text file using vaderSentiment library
- Write the sentiment scores to a CSV file
- Called by: main.py or as standalone program

### **Libraries**

| Library | Description |
| --- | --- |
| textblob | Natural language processing library that provides a simple API for common NLP tasks such as sentiment analysis and text classification. |
| feature_extraction | Library for extracting features from text data. |
| text_preprocessing | Library for preprocessing text data, such as cleaning and tokenizing. |
| sys | Library for interacting with the Python runtime system. |
| sklearn | Machine learning library that provides a wide range of supervised and unsupervised learning algorithms. |
| torch | Machine learning library that provides a fast and flexible deep learning framework. |
| spacy | Natural language processing library that provides a high-performance statistical NLP engine. |
| csv | Library for reading and writing CSV files. |
| gensim | Natural language processing library that provides a number of tools for working with large-scale text corpora. |
| thefuzz | Library for fuzzy string matching. |
| threading | Library for creating and managing threads. |
| transformers | Library for natural language processing that provides a number of pre-trained models for tasks such as text classification and question answering. |
| numpy | Library for scientific computing that provides a high-performance multidimensional array object. |
| chardet | Library for detecting the encoding of a text file. |
| stanza | Natural language processing library that provides a high-performance statistical NLP engine for the Turkish language. |
| glob | Library for finding files that match a pattern. |
| google_auth_oauthlib | Library for authenticating with Google APIs. |
| pdf2image | Library for converting PDF files to images. |
| sumy | Library for summarizing text documents. |
| info_extraction | Library for extracting information from text documents, such as entities, dates, and keywords. |
| time | Library for getting the current time and date. |
| warnings | Library for handling warnings. |
| flair | Natural language processing library that provides a number of pre-trained models for tasks such as sentiment analysis and named entity recognition. |
| vaderSentiment | Library for sentiment analysis. |
| concurrent | Library for managing concurrent execution of tasks. |
| google | Library for interacting with Google APIs. |
| fuzzywuzzy | Library for fuzzy string matching. |
| manage_files | Library for working with files. |
| PyPDF4 | Library for reading and writing PDF files. |
| re | Library for regular expressions. |
| os | Library for interacting with the operating system. |
| openai | Library for interacting with OpenAI APIs. |
| base64 | Library for encoding and decoding binary data in base64 format. |
| nltk | Natural language processing library that provides a wide range of NLP tools and resources. |
| pandas | Library for data analysis and manipulation. |
| similarity_clustering | Library for clustering and finding similar items in a dataset. |
| functools | Library for providing higher-order functions and other tools for working with functions. |
| requests | Library for making HTTP requests. |
| googleapiclient | Library for interacting with Google APIs. |
| rake_nltk | Library for natural language processing that provides a number of tools for extracting keywords from text. |
| subprocess | Library for launching subprocesses. |
| fitz | Library for working with PDF files. |
| pytesseract | Library for optical character recognition. |
| collections | Library for providing a number of container data types, such as lists, sets, and dictionaries.
