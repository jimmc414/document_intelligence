import gensim.downloader as api

# Load pre-trained word2vec model
model = api.load("word2vec-google-news-300")

def vectorize_text(tokens):
    vectors = [model[token] for token in tokens if token in model]
    if len(vectors) > 0:
        avg_vector = sum(vectors) / len(vectors)
        return avg_vector
    else:
        return None