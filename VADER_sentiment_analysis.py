import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the SentimentIntensityAnalyzer from Vader
analyzer = SentimentIntensityAnalyzer()

def sentiment_analysis(text):
    # Calculate sentiment scores using the analyzer
    scores = analyzer.polarity_scores(text)
    return scores

def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)
            with open(file_path, "r", encoding="utf-8") as input_file:
                text = input_file.read()

            results = sentiment_analysis(text)

            output_file_path = os.path.join(output_dir, f"VADER_{file}")

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for key, value in results.items():
                    output_file.write(f"{key}: {value}\n")

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\VADER_sentiment"
os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)


# This code uses the VaderSentiment library to perform sentiment analysis on the text files in the `txt_output` directory and writes the output to the `VADER_sentiment` folder with filenames prepended with `VADER_`. The Vader sentiment analysis provides a compound score representing the overall sentiment, as well as individual scores for positive, negative, and neutral sentiment.