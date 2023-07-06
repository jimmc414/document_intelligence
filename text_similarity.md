# Documentation

## Setup

Before running this Python program, you need to ensure that you have the necessary environment and dependencies set up:

### Python

If you haven't installed Python yet, navigate to the official Python downloads page at `https://www.python.org/downloads/` and get the latest Python version suitable for your OS.

### Required Packages

This script requires the following Python packages: `os`, `gensim`, `numpy`, `scipy`, `nltk`, `configparser`, `pickle` and `shutil`. Most of these packages come by default with a standard Python installation, however, to ensure they are installed and updated, run the following command in your terminal:

```
pip install gensim numpy scipy nltk configparser
```

### NLTK Data

For the NLTK package to work properly, you need to download some additional data. Run these two commands in your Python environment:

```python
import nltk
nltk.download('punkt')  # Punkt Tokenizer Model
nltk.download('stopwords')  # Stopwords Corpus
```

### Configuration File

Prepare a settings.ini file which has the directory path that holds the text documents. The ini file should look like this:

```ini
[paths]
txt_documents = /path/to/your/documents
```

This ini file must be in the same directory as your Python script.

## Tutorial

This Python program is used for finding similarity between different text documents. It uses `gensim's Word2Vec` which converts text into numeric vectors. Here are the steps to follow:

1. Run the script.
2. You will be prompted to enter the name of the text file.
3. Enter a similarity threshold value between 0 and 1.
4. Decide whether to move the similar text files to another subfolder. 

All text files with similarity scores above the threshold will be printed out and written to `output.txt`. If required, files can be moved to a subfolder within your documents folder. 

## Explanation

When you first run the script, it checks if `Word2Vec ('word2vec-google-news-300')` is downloaded or not. If not, it downloads and saves the model into a pickle file for future use. The input text file and all `.txt` files in the documents folder are read, preprocessed, and then converted to vectors. Cosine similarity is computed between the input file's vector and each document's vector. 

## Reference

The key functions include:

- `preprocess(text)`: tokenizes, convert to lower case, removes stop words and non-alphabetic characters.
- `get_feature_vec(words, model)`: generates the feature vector for the given words using the Word2Vec model.
- `compute_similarity(vec1, vec2)`: calculates the cosine similarity between two vectors.
- `print_report(report, title)`: it formats and prints out the similarity report in a readable way.

## How-to Guides

When choosing similarity threshold, higher threshold means getting files that are very similar to the input file. 

## Troubleshooting

For errors:
- Check if settings.ini file is correctly configured and exists in the same directory as the Python script.
- Ensure the Word2Vec model file is properly downloaded and saved.
- Check the existence and contents of your text files in the specified directory. 