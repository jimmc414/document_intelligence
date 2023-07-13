
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

This Python program is designed to find similarities between different text documents using Word2Vec model to convert text into numeric vectors. Here are the steps to follow:

1. Run the script.
2. You will be prompted to enter the name of the text file you want to compare with others.
3. Enter a similarity threshold value between 0 and 1.
4. Decide whether to move the similar text files to another subfolder.

All text files with similarity scores above the threshold will be printed out and written to an output file named `output.txt`. If requested, files can be moved to a subfolder within your documents folder.

## Explanation

The program operates in the following order:

1. Reads configuration settings from a `settings.ini` file.
2. Prompts the user for the name of a text file to compare with others, a similarity threshold, and whether similar files should be moved to a subfolder.
3. Tokenizes and removes stop words from the text using the Natural Language Toolkit (NLTK).
4. Converts the preprocessed text into vectors using the Word2Vec model.
5. Computes the cosine similarity between the vectors of the input file and each document in the corpus.
6. Filters out the files that have a similarity score above the user-specified threshold and writes the filtered similarity report to an output text file.
7. If the user has opted to move similar files to a subfolder, the script will do so.
8. If the Word2Vec model isn't already present, the program will download it via the gensim API and cache it using pickle for future use.

## Reference

The key functions include:

- `preprocess(text)`: Tokenizes the text, converts it to lower case, removes stop words, and discards non-alphabetic characters.
- `get_feature_vec(words, model)`: Generates the feature vector for the given words using the Word2Vec model.
- `compute_similarity(vec1, vec2)`: Calculates the cosine similarity between two vectors.
- `print_report(report, title)`: Formats and prints out the similarity report.

## How-to Guides

When choosing a similarity threshold, a higher threshold will yield files that are very similar to the input file.

## Troubleshooting

For errors:

- Check if the settings.ini file is correctly configured and exists in the same directory as the Python script.
- Ensure the Word2Vec model file is properly downloaded and saved.
- Check the existence and contents of your text files in the specified directory. 
