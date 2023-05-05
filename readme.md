---

# AutoIndex: PDF Text Analysis and Processing

AutoIndex is a Proof of Concept designed to automatically process PDF files. It extracts relevant information, analyzes sentiment, clusters, and categorizes documents based on similarity, generates summaries, and extracts text.

This information will be used to automatically assign the correct account number in an external system.

This project aims to automatically digest unstructured documents, phone calls, or texts and generate metadata that indexes the correct account in a remote system. The objective is for this process to occur end-to-end, from unknown document to correctly filed, without human intervention.

The project consists of several Python scripts that accomplish different tasks, including OCR processing, named entity recognition, similarity clustering, sentiment analysis, and text summarization.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- PyMuPDF
- pdf2image
- PyTesseract
- NLTK
- Spacy
- Gensim
- TextBlob
- Sumy
- Scikit-learn
- Transformers (Hugging Face)
- OpenAI API (For Audio transcription)

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