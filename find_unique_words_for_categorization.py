import os
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

folder_path = "C:\\python\\autoindex\\txt_output"

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
documents = []

for file in text_files:
    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
        text = f.read()
        preprocessed_text = preprocess(text)
        documents.append(preprocessed_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Use K-means clustering to group similar documents
num_clusters = 5  # Change this number based on how many groups you want
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(tfidf_matrix)

clusters = {}
for i, cluster in enumerate(km.labels_):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append((text_files[i], tfidf_matrix[i]))

results = {}
for i, (file, cluster) in enumerate(zip(text_files, km.labels_)):
    top_words = pd.Series(data=tfidf_matrix.toarray()[i], index=vectorizer.get_feature_names_out()).sort_values(ascending=False).head(30)
    document_vector = tfidf_matrix[i].reshape(1, -1)
    centroid_vector = km.cluster_centers_[cluster].reshape(1, -1)
    similarity = cosine_similarity(document_vector, centroid_vector)
    results[file] = {"cluster": cluster, "similarity": similarity[0][0], "top_words": top_words.index.tolist()}

# Sort documents within each cluster by similarity scores
for cluster in clusters:
    clusters[cluster] = sorted(clusters[cluster], key=lambda x: cosine_similarity(x[1].reshape(1, -1), km.cluster_centers_[cluster].reshape(1, -1)), reverse=True)

with open("unique_words_for_categorization.txt", "w", encoding="utf-8") as f:
    for cluster in clusters:
        f.write(f"Cluster {cluster}:\n")
        for item in clusters[cluster]:
            file = item[0]
            f.write(f"  Document: {file}\n")
            f.write(f"  Similarity: {results[file]['similarity']:.4f}\n")
            f.write("  Top 30 unique words:\n")
            for word in results[file]['top_words']:
                f.write(f"  {word}\n")
            f.write("\n")