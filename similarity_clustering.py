from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter

def determine_best_similarity_measure():
    return "cosine"

def optimal_clusters(x):
    return 3

def extract_specific_values(texts, specific_phrases):
    extracted_texts = []
    for text in texts:
        extracted_text = []
        for phrase in specific_phrases:
            if phrase.lower() in text.lower():
                extracted_text.append(phrase.lower())
        extracted_texts.append(" ".join(extracted_text))
    return extracted_texts

def get_most_frequent_phrase(cluster_labels, texts, specific_phrases):
    phrase_count_list = [Counter() for _ in range(max(cluster_labels) + 1)]
    for label, text in zip(cluster_labels, texts):
        for phrase in specific_phrases:
            if phrase.lower() in text.lower():
                phrase_count_list[label][phrase.lower()] += 1
    most_frequent_phrases = []
    for phrase_count in phrase_count_list:
        most_freq_phrase = phrase_count.most_common(1)
        if most_freq_phrase:
            most_frequent_phrases.append(most_freq_phrase[0][0])
        else:
            most_frequent_phrases.append("")
    return most_frequent_phrases

def cluster_analysis(texts, specific_phrases):
    cluster_algorithm = KMeans
    similarity_measure = determine_best_similarity_measure()
    n_clusters = optimal_clusters(texts)
    
    extracted_texts = extract_specific_values(texts, specific_phrases)
    vectorizer = TfidfVectorizer(analyzer='word', lowercase=True, use_idf=True, min_df=2)
    non_empty_texts = [text for text in extracted_texts if text.strip()]
    if not non_empty_texts:
        return None, None
    X = vectorizer.fit_transform(non_empty_texts)
    
    if not vectorizer.vocabulary_:
        return None, None
    
    cluster_model = cluster_algorithm(n_clusters=n_clusters)
    cluster_labels = cluster_model.fit_predict(X)
    
    most_frequent_phrases = get_most_frequent_phrase(cluster_labels, texts, specific_phrases)
    
    return cluster_labels, most_frequent_phrases