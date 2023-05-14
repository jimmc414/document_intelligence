
"A business is simply an idea to make other people's lives better."

—Sir Richard Branson

---

# AutoIndex: PDF Text Analysis and Processing
### Description
AutoIndex is a document intelligence proof of concept that is designed to automatically process PDF files. It extracts relevant information, analyzes sentiment, clusters and categorizes documents based on similarity, and generates summaries. The information is used to automatically assign the correct account number to an external system. The project aims to automatically digest unstructured documents, such as phone calls and texts, and generate metadata indexes for the correct account in a remote system. The objective is for this process to occur end-to-end, from an unknown document to being correctly filed, without human intervention.

The project consists of several Python scripts that accomplish different tasks, including OCR processing, named entity recognition, similarity clustering, sentiment analysis, text summarization, audio transcription, email extraction, key-value pair extraction, and classification.

### Installation / How to setup
To run this project, you need to have Python 3.x installed on your system. You also need to install the following libraries and tools:

- PyMuPDF: A Python library that enables access and manipulation of PDF files, including text extraction, metadata retrieval, layout analysis.
pdf2image: A Python library that converts PDF files to images, allowing easier processing and analysis of non-textual content in the files.
- PyTesseract: A Python wrapper for the Tesseract OCR engine, which enables optical character recognition (OCR) capabilities for converting scanned images or PDFs to text.
- NLTK: The Natural Language Toolkit (NLTK) is a Python library for working with human language data, providing tools for text processing, tokenization, parsing, classification, etc.
- spaCy: A fast and modern Python library for natural language processing (NLP), offering pretrained models and extensive functionality for NLP tasks such as text tokenization, part-of-speech tagging, named entity recognition, etc.
- Gensim: A Python library specialized in topic modeling and document similarity analysis, which allows efficient large-scale text processing using algorithms like Word2Vec, FastText, Latent Semantic Analysis (LSA), etc.
- TextBlob: A simple Python library for processing textual data, offering basic NLP features such as part-of-speech tagging, noun phrase extraction, sentiment analysis, translation, etc.
- Sumy: A Python library that provides functionality to summarize texts using various algorithms, extracting key information from long articles or documents.
- Scikit-learn: A popular Python library for machine learning and data science, offering tools for preprocessing data, training algorithms, evaluating model performance, etc.
- Transformers (Hugging Face): A Python library for state-of-the-art natural language processing (NLP), based on transformer model architectures like BERT and GPT-3. It offers pre-trained models for tasks like text generation, question-answering (QA), sentiment analysis (SA), etc.
- OpenAI API - Whisper: An API service provided by OpenAI that offers an automatic speech recognition (ASR) system called Whisper. It transcribes spoken language audio data to written text with high accuracy. It requires an OpenAI API key to use.

You can use pip to install the required packages in your Python environment:
```
pip install pymupdf pdf2image pytesseract nltk spacy gensim textblob sumy scikit-learn transformers openai os
```
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

@misc{autoindex,
  author = {Your Name},
  title = {AutoIndex: PDF Text Analysis and Processing},
  year = {2023},
  howpublished = {\url{https://github.com/yourusername/autoindex}}
}

### Other related tools
Here are some other related tools or resources that are relevant to document intelligence or text analysis:

Apache Tika: A Java library that detects and extracts metadata and text from over a thousand different file types (such as PPT, XLS, and PDF).
Stanford CoreNLP: A Java suite of core NLP tools that provides a set of natural language analysis tools which can take raw English language text input and give various forms of linguistic analysis output.
BERT Extractive Summarizer: A Python library that uses a pre-trained BERT model to extract summaries from long texts.
pdfminer: A Python tool for extracting information from PDF documents. Unlike other PDF-related tools, it focuses entirely on getting and analyzing text data.
pdfplumber: A Python library that provides a set of tools for extracting information from PDF files. It allows access to both visual elements (such as tables) and textual elements (such as headers).