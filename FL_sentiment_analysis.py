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