import os 
import gensim

# Create a dictionary object that maps words to ids.
dictionary = gensim.corpora.Dictionary()

# Load the pre-trained topic model
model = gensim.models.ldamodel.LdaModel.load('model_all_no_lemma')

def topic_modeling(text):
    # Convert the text to a list of words
    words = text.split()

    # Convert the words to a list of (word_id, word_count) tuples using the dictionary
    bow = dictionary.doc2bow(words)

    # Predict topics using the model
    topics = model.get_document_topics(bow)

    # Return topics as a string
    return "\n".join(str(t) for t in topics)

def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)
            with open(file_path, "r", errors="ignore") as input_file:
                text = input_file.read()

            results = topic_modeling(text)

            output_file_path = os.path.join(output_dir, f"TM_{file}")

            with open(output_file_path, "w") as output_file:
                output_file.write(results)

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\TM_topics"
os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)


# Training your own LDA model with Gensim is not very hard, but it requires some steps. Here is a brief guide based on 1:

# Load and preprocess your text documents. You can use nltk or spacy to tokenize, remove stopwords, lemmatize, etc.
# Split the documents into tokens and create a Dictionary object that maps words to ids using gensim.corpora.Dictionary.
# Convert the documents to a list of (word_id, word_count) tuples using the dictionary’s doc2bow method. This is your corpus.
# Train the LDA model on the corpus using gensim.models.LdaModel. You can specify the number of topics, the alpha parameter, the number of passes, etc.
# Save the model to disk using the save method or load a pre-trained model using the load method.
# For example:

# from gensim import corpora, models
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords

# # Load and preprocess text documents
# docs = ... # list of strings
# tokenizer = RegexpTokenizer(r'\w+')
# stop_words = stopwords.words('english')
# texts = [[token for token in tokenizer.tokenize(doc.lower()) if token not in stop_words] for doc in docs]

# # Create dictionary and corpus
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]

# # Train LDA model
# model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=5)

# # Save or load model
# model.save('lda.model')
# model = models.LdaModel.load('lda.model')