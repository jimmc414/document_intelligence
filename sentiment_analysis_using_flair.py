import os
from flair.models import TextClassifier
from flair.data import Sentence

# Load the pre-trained sentiment analysis model
model = TextClassifier.load('en-sentiment')

def sentiment_analysis(text):
    # Create a Sentence object from input text
    sentence = Sentence(text)

    # Predict sentiment using the model
    model.predict(sentence)

    # Return label and score if it exists, otherwise return "Unknown"
    if sentence.labels:
        sentiment = sentence.labels[0]
        return str(sentiment)
    else:
        return 'Unknown'

def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)
            with open(file_path, "r", encoding="utf-8") as input_file:
                text = input_file.read()

            results = sentiment_analysis(text)

            output_file_path = os.path.join(output_dir, f"FL_{file}")

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(results)

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\FL_sentiment"
os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)


# To customize the Flair emotion classifier with your own data and labels, you can follow these steps:

# Prepare your data in a CSV file with two columns: text and label. The text column should contain the sentences you want to classify, and the label column should contain the emotion labels you want to use. For example:
# text	label
# I am so happy that we won the case	joy
# I am so angry that we lost the case	anger
# I am so scared that we will lose the case	fear
# I am so sad that we settled the case	sadness
# Split your data into three files: train.csv, dev.csv and test.csv. The train.csv file should contain the majority of your data (e.g. 80%) that you will use to train your model. The dev.csv file should contain a smaller portion of your data (e.g. 10%) that you will use to validate your model during training. The test.csv file should contain another smaller portion of your data (e.g. 10%) that you will use to evaluate your model after training. Make sure that each file has the same format and columns as the original data file.
# Create a Corpus object from your data files using Flair’s CSVClassificationCorpus class. This object will load your data and create a dictionary of labels for your model. For example:
# # Import Flair
# import flair

# # Create a Corpus object from your data files
# corpus = flair.datasets.CSVClassificationCorpus(
    # data_folder='path/to/your/data/folder', # The folder where you saved your data files
    # column_name_map={0: 'text', 1: 'label'}, # A dictionary that maps column indices to names
    # skip_header=True # Whether to skip the first row of the CSV file
# )
# 
# Create a TextClassifier object from a pre-trained embedding model and a MultiLabelClassificationHead. The embedding model will encode your sentences into numerical vectors, and the classification head will predict the emotion labels from the vectors. You can choose any pre-trained embedding model from Flair’s list of available models, such as ‘bert-base-uncased’ or ‘roberta-base’. For example:
# # Import Flair
# import flair

# # Create a TextClassifier object
# classifier = flair.models.TextClassifier(
    # embeddings=flair.embeddings.TransformerWordEmbeddings('bert-base-uncased'), # A pre-trained embedding model
    # label_dictionary=corpus.make_label_dictionary(), # A dictionary of labels from your corpus
    # multi_label=False # Whether to allow multiple labels per sentence or not
# )
# 
# Create a ModelTrainer object from your classifier and corpus. This object will train your classifier on your corpus using various parameters and settings. For example:
# # Import Flair
# import flair

# # Create a ModelTrainer object
# trainer = flair.trainers.ModelTrainer(
    # model=classifier, # Your classifier object
    # corpus=corpus # Your corpus object
# )
# 
# Train your classifier using the train method of the ModelTrainer object. You can specify various parameters and settings for the training process, such as the number of epochs, the learning rate, the batch size, etc. For example:
# # Train your classifier
# trainer.train(
    # base_path='path/to/your/model/folder', # The folder where you want to save your trained model
    # max_epochs=10, # The maximum number of epochs to train for
    # learning_rate=0.01, # The learning rate for the optimizer
    # mini_batch_size=32, # The batch size for each iteration
    # monitor_test=True # Whether to evaluate your model on the test set after each epoch or not
# )

# Evaluate your classifier using the final_test_results.json file in your model folder. This file will contain various metrics and scores for your model’s performance on the test set, such as accuracy, precision, recall, F1-score, etc. For example:
# {
  # "test_score": 0.9,
  # "test_precision": 0.91,
  # "test_recall": 0.89,
  # "test_f1-score": 0.9,
  # "test_loss": 0.23,
  # "detailed_test_metrics_per_label": {
    # "joy": {
      # "precision": 0.92,
      # "recall": 0.88,
      # "f1-score": 0.9,
      # "support": 50
    # },
    # "anger": {
      # "precision": 0.9,
