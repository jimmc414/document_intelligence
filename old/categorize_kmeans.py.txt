# Import libraries
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Define some helper functions

def read_files(path):
    # Read all text files from a given path and return a list of file names and contents
    file_names = []
    file_contents = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_names.append(file)
            with open(os.path.join(path, file), encoding="utf8", errors="ignore") as f:
                file_contents.append(f.read())
    return file_names, file_contents

def preprocess(text):
    # Preprocess a given text by removing stopwords, punctuation, numbers, and stemming
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    text = text.lower() # convert to lower case
    text = re.sub(r"[^\w\s]", "", text) # remove punctuation
    text = re.sub(r"\d+", "", text) # remove numbers
    text = [stemmer.stem(word) for word in text.split() if word not in stop_words] # remove stopwords and stem words
    text = " ".join(text) # join words back to text
    return text

def print_clusters(model, feature_names, n_top_words):
    # Print the cluster labels and the top words for each cluster
    print("Cluster labels:")
    print(model.labels_)
    print()
    print("Top words per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    for i in range(model.n_clusters):
        print("Cluster %d:" % i, end="")
        for ind in order_centroids[i, :n_top_words]:
            print(" %s" % feature_names[ind], end="")
        print()
        
def save_clusters(file_names, labels, path):
    # Save the cluster labels to a text file with the same filename in a given path
    for i in range(len(file_names)):
        file_name = file_names[i]
        label = labels[i]
        with open(os.path.join(path, file_name), "w") as f:
            f.write(str(label))

# Define some parameters

input_path = "C:\\python\\autoindex\\txt_output" # input directory for text files
output_path = "C:\\python\\autoindex\\category" # output directory for cluster labels
n_components = 10 # number of topics to extract using LSA
n_clusters = 5 # number of clusters to form using K-means
n_top_words = 10 # number of top words to display for each cluster

# Read the text files from the input directory

file_names, file_contents = read_files(input_path)

# Create a pandas dataframe to store the file names and contents

data = pd.DataFrame({"file_name": file_names, "file_content": file_contents})

# Preprocess the file contents

data["file_content"] = data["file_content"].apply(preprocess)

# Create a document-term matrix using TF-IDF vectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data["file_content"])

# Apply LSA using truncated SVD to reduce the dimensionality and extract latent topics

svd = TruncatedSVD(n_components=n_components)
X_topics = svd.fit_transform(X)

# Cluster the documents based on their topic scores using K-means

km = KMeans(n_clusters=n_clusters, random_state=0)
km.fit(X_topics)

# Print the cluster labels and the top words for each cluster to the console

feature_names = vectorizer.get_feature_names_out()
print_clusters(km, feature_names, n_top_words)

# Save the cluster labels to a text file with the same filename in the output directory

save_clusters(data["file_name"], km.labels_, output_path)