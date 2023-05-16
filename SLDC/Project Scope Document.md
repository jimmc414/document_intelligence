Project Scope Document

Project Name: Automated Document Categorization and Information Extraction System

Introduction:

This project aims to develop an automated document categorization and information extraction system that can efficiently process a set of given documents in PDF format, extract relevant information, and classify them. The system will also analyze sentiments and generate summaries for the documents. The project covers various tasks including text extraction, OCR processing, named entity recognition, similarity clustering, sentiment analysis, and text summarization.

Project Goals:

1\. Extract searchable text from PDF files and convert non-searchable PDF files using OCR.

2\. Extract named entities such as proper names, case numbers, account numbers, and legal terms from the extracted text.

3\. Categorize and cluster documents based on their similarity.

4\. Perform sentiment analysis on the extracted text.

5\. Generate text summarizations for the extracted text using different summarization algorithms.

Components:

1\. Text Extraction from PDF Files (autoextractpdf2text.py)

    - Check for searchable PDF files and extract text using the PyMuPDF library.

2\. OCR Processing for Non-searchable PDF Files (autoocr_parallel.py)

    - Convert non-searchable PDFs to text using pytesseract and pdf2image libraries.

3\. Named Entity Recognition (autoner.py)

    - Extract proper names, case numbers, account numbers, and legal terms using the Spacy library.

4\. Similarity Clustering (automatic_similarity_clustering.py)

    - Determine the best similarity measure and cluster documents based on their similarity using the Scikit-learn library.

5\. Sentiment Analysis (autosentiment.py)

    - Perform sentiment analysis on the extracted text using the TextBlob library.

6\. Text Summarization (autosummarize.py & pegasus_summarize.py)

    - Generate summaries of the extracted text using the Sumy library and Pegasus model from the Hugging Face Transformers library.

Project Deliverables:

1\. Python scripts for each component.

2\. A main.py script to run the complete process in a sequential manner.

3\. A project directory structure containing input files, output files, and scripts.

4\. Documented source code in a publicly accessible repository.

5\. A README.md file that provides information on the project structure, instructions to run the project, and prerequisites for the packages required.

Assumptions:

1\. Input data is in the form of PDF files.

2\. Python 3.x is installed on the system.

3\. All prerequisite packages are installed.

Constraints:

1\. Processing large volumes of PDF files may require more time and computational resources.

2\. Export restrictions on certain Natural Language Processing (NLP) libraries.

Success Factors:

1\. Efficient extraction of searchable text from PDF files.

2\. Accurate OCR processing for non-searchable PDF files.

3\. Proper categorization and clustering of documents based on similarity.

4\. Accurate sentiment analysis and summarization of extracted text.

Project Timeline:

1\. Research and initial setup: 2 weeks

2\. Implementation of individual components: 4 weeks

3\. Integration and testing across components: 2 weeks

4\. Documentation and packaging: 1 week

5\. Buffer for revisions and bug-fixes: 1 week

   Total project duration: 10 weeks