# Document Intelligence

A comprehensive Python-based document processing toolkit for OCR, text extraction, NLP analysis, and document classification.

## Features

- **OCR Processing**: Extract text from PDF documents using Tesseract OCR
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- **Named Entity Recognition**: Extract persons, organizations, locations, and custom entities
- **Sentiment Analysis**: Multiple engines (TextBlob, Flair, VADER, HuggingFace)
- **Document Similarity**: Compare documents using Word2Vec, TF-IDF, and GZIP-based methods
- **Document Clustering**: Group similar documents using K-means and LSA
- **Text Summarization**: Automatic text summarization using LSA
- **Document Classification**: Classify documents into categories
- **Email Processing**: Download and process emails from Gmail
- **Key-Value Extraction**: Extract structured data from documents

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for optical character recognition)
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **macOS**: `brew install tesseract`
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Download Required NLP Models

#### spaCy Model
```bash
python -m spacy download en_core_web_sm
```

#### NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### TextBlob Corpora
```bash
python -m textblob.download_corpora
```

## Configuration

### Optional: Create settings.ini

Copy the example configuration file and customize it:

```bash
cp settings.ini.example settings.ini
```

Edit `settings.ini` to configure:
- Document paths
- Similarity thresholds
- Document categories for classification

**Note**: If `settings.ini` is not found, scripts will use sensible defaults.

### Optional: Gmail API Setup

For email downloading features, you'll need Google API credentials:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Download the credentials file as `credentials.json` in the project root

## Project Structure

```
document_intelligence/
├── documents/              # Input PDF documents
├── txt_output/            # Extracted text files
├── category/              # Clustered documents
├── NER/                   # Named entity extraction results
├── sentiments/            # Sentiment analysis results
├── summarization/         # Document summaries
├── document_classification/ # Classification results
├── FL_sentiment/          # Flair sentiment analysis results
├── kvextract/             # Key-value extraction results
└── extract/               # Pattern extraction results
```

## Usage

### Main Pipeline

Run the main document processing pipeline:

```bash
python main.py
```

This will:
1. Process PDFs with OCR
2. Extract and preprocess text
3. Generate document vectors
4. Cluster similar documents

### Individual Scripts

#### OCR Processing
```bash
python optical_character_recognition.py document1.pdf document2.pdf
```

#### Sentiment Analysis
```bash
python sentiment_analysis.py                    # TextBlob
python sentiment_analysis_using_flair.py        # Flair
python sentiment_analysis_using_vader.py        # VADER
```

#### Named Entity Recognition
```bash
python extract_named_entities.py
```

#### Document Clustering
```bash
python cluster_documents.py
python fuzzy_categorize_documents.py
```

#### Text Summarization
```bash
python summarize_text.py
```

#### Document Similarity
```bash
python document_similarity.py
python text_similarity.py
python gzip_knn_similarity.py
```

#### Email Processing
```bash
python download_email.py
python dl_email.py
```

## Bug Fixes (Latest Release)

This release includes comprehensive bug fixes that resolve all execution-blocking issues:

### Critical Fixes
- ✅ Fixed syntax error in `optical_character_recognition.py` (en-dash → hyphen in Tesseract config)
- ✅ Added missing `logging` import in `optical_character_recognition.py`
- ✅ Fixed module-level model downloads in `extract_features_from_text.py` and `sentiment_analysis_using_flair.py` (now uses lazy loading)
- ✅ Added all missing dependencies to `requirements.txt`:
  - google-auth, google-auth-oauthlib, google-api-python-client
  - textblob, flair, sumy, fuzzywuzzy
  - scikit-learn, pandas, torch, transformers

### Platform Compatibility
- ✅ Replaced all hardcoded Windows paths with cross-platform `os.path.join()`
- ✅ Made Tesseract path platform-aware (Windows vs Linux/Mac)
- ✅ All output directories now created automatically with `os.makedirs(exist_ok=True)`

### Configuration & Error Handling
- ✅ Graceful handling of missing `settings.ini` (uses sensible defaults)
- ✅ Graceful handling of missing `credentials.json` (clear error message with instructions)
- ✅ Created `settings.ini.example` template for easy configuration

### Type Errors & Logic Bugs
- ✅ Fixed type error in `document_classification.py` (convert numpy array to string)
- ✅ Fixed logic error in `document_similarity.py` (now reads file contents instead of comparing file paths)
- ✅ Fixed newline escaping in `gzip_knn_similarity.py` (`\\n` → `\n`)
- ✅ Fixed missing output directory creation in multiple scripts

### Code Quality Improvements
- ✅ Moved module-level script logic into `main()` functions
- ✅ Added `if __name__ == "__main__"` guards to prevent execution on import
- ✅ Optimized spaCy model loading (load once at module level instead of per function call)

## Dependencies

See `requirements.txt` for the complete list. Major dependencies include:

- **NLP**: spaCy, NLTK, Flair, TextBlob, Gensim, Transformers
- **ML**: scikit-learn, PyTorch, pandas, numpy, scipy
- **OCR**: pytesseract, pdf2image, PyMuPDF, PyPDF4, Pillow
- **Other**: google-api-python-client, fuzzywuzzy, sumy

## Performance Notes

- First run may take longer due to model downloads (Word2Vec, Flair, etc.)
- Models are cached after first download
- Word2Vec model (~1.6GB) is downloaded on-demand when needed
- Use `--verbose` flag (where available) for detailed progress

## Troubleshooting

### Tesseract Not Found
Ensure Tesseract is installed and in your system PATH, or edit the path in `optical_character_recognition.py`

### spaCy Model Not Found
Run: `python -m spacy download en_core_web_sm`

### Gmail API Errors
Ensure `credentials.json` is present and you've enabled the Gmail API in Google Cloud Console

### Out of Memory
For large document sets, process in smaller batches or increase system RAM

## Contributing

Contributions are welcome! Please ensure all code:
- Uses cross-platform paths (`os.path.join()`)
- Includes error handling
- Uses lazy loading for large models
- Has proper documentation

## License

[Add your license here]

## Support

For issues and questions, please open an issue on the project repository.
