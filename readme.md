
"A business is simply an idea to make other people's lives better."

—Sir Richard Branson

---

# Document Intelligence: Document Text Analysis and Artificial Intelligence Processing
### Description
Document Intelligence is a document intelligence proof of concept that is designed to automatically process PDF files. It extracts relevant information, analyzes sentiment, clusters and categorizes documents based on similarity, and generates summaries. The information is used to automatically assign the correct account number to an external system. The project aims to automatically digest unstructured documents, such as phone calls and texts, and generate metadata indexes for the correct account in a remote system. The objective is for this process to occur end-to-end, from an unknown document to being correctly filed, without human intervention.

The project consists of several Python scripts that accomplish different tasks, including OCR processing, named entity recognition, similarity clustering, sentiment analysis, text summarization, audio transcription, email extraction, key-value pair extraction, and classification.

### Installation / How to setup

You can use pip to install the required packages in your Python environment:
```
pip install textblob feature_extraction text_preprocessing sys sklearn torch spacy csv gensim thefuzz threading transformers numpy chardet stanza glob google_auth_oauthlib pdf2image sumy info_extraction time warnings flair vaderSentiment concurrent google fuzzywuzzy file_utils PyPDF4 re os openai base64 nltk pandas similarity_clustering functools requests googleapiclient rake_nltk subprocess fitz pytesseract collections
```

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
| file_utils | Library for working with files. |
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


You also need to download the necessary language model for spaCy:
```
python -m spacy download en_core_web_sm
```
Additionally, you need to install Poppler and Tesseract on your system and set the necessary dependencies for the Python project. To do this, follow these steps:

Download and install Poppler from https://pypi.org/project/pdf2image/. There is an up-to-date version for Windows users.

Move the extracted directory to a desired location on your system.

Add the bin/ directory to your path environment variable.

Alternatively, you can use the poppler_path = r"c:\path\to\poppler-xx\bin" argument in the convert_from_path function in the autoextractpdf2text.py script.

Download and install Tesseract OCR from its official website (https://github.com/ub-mannheim/tesseract/wiki).

After installation, add the Tesseract OCR executable to your system’s path. For Windows, the default installation path is c:\program files\tesseract-ocr\tesseract.exe. Add this path to your system’s path variable or configure the PyTesseract library to use the installed Tesseract directly.

Verify your Tesseract installation: In the Python script.
```
import pytesseract
print(pytesseract.get_tesseract_version())
```
You should see something like tesseract 4.1.1 or higher.

Copy-pastable quick start code example
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

### Recommended citation
If you use this project for your own research or work, please cite it as follows:

@misc{Document Intelligence,
  author = {Your Name},
  title = {Document Intelligence: PDF Text Analysis and Processing},
  year = {2023},
  howpublished = {\url{https://github.com/yourusername/Document Intelligence}}
}

### Other related tools
Here are some other related tools or resources that are relevant to document intelligence or text analysis:

Apache Tika: A Java library that detects and extracts metadata and text from over a thousand different file types (such as PPT, XLS, and PDF).
Stanford CoreNLP: A Java suite of core NLP tools that provides a set of natural language analysis tools which can take raw English language text input and give various forms of linguistic analysis output.
BERT Extractive Summarizer: A Python library that uses a pre-trained BERT model to extract summaries from long texts.
pdfminer: A Python tool for extracting information from PDF documents. Unlike other PDF-related tools, it focuses entirely on getting and analyzing text data.
pdfplumber: A Python library that provides a set of tools for extracting information from PDF files. It allows access to both visual elements (such as tables) and textual elements (such as headers).