import gensim.downloader as api

# Cache for the model to avoid reloading
_model = None

def _get_model():
    """Lazily load the word2vec model only when needed."""
    global _model
    if _model is None:
        print("Loading word2vec-google-news-300 model... (this may take a few minutes on first run)")
        _model = api.load("word2vec-google-news-300")
    return _model

def vectorize_text(tokens):
    model = _get_model()
    vectors = [model[token] for token in tokens if token in model]
    if len(vectors) > 0:
        avg_vector = sum(vectors) / len(vectors)
        return avg_vector
    else:
        return None
