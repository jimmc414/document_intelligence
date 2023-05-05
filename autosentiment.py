import os
import sys
import nltk
import file_utils
from textblob import TextBlob

def sentiment_analysis(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as input_file:
        text = input_file.read()

    blob = TextBlob(text)
    sentiment = blob.sentiment

    polarity_interpretation = ""
    if sentiment.polarity < -0.5:
        polarity_interpretation = "highly negative"
    elif sentiment.polarity < 0:
        polarity_interpretation = "slightly negative"
    elif sentiment.polarity == 0:
        polarity_interpretation = "neutral"
    elif sentiment.polarity <= 0.5:
        polarity_interpretation = "slightly positive"
    else:
        polarity_interpretation = "highly positive"

    subjectivity_interpretation = ""
    if sentiment.subjectivity < 0.5:
        subjectivity_interpretation = "somewhat objective"
    else:
        subjectivity_interpretation = "somewhat subjective"

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"Polarity: {sentiment.polarity} ({polarity_interpretation})\n")
        output_file.write(f"Subjectivity: {sentiment.subjectivity} ({subjectivity_interpretation})\n")

if __name__ == "__main__":
    input_directory = "c:\\python\\autoindex\\txt_output"
    output_directory = "c:\\python\\autoindex\\sentiments"
    os.makedirs(output_directory, exist_ok=True)

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(".txt"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(
                    output_directory, os.path.splitext(file)[0] + "_sentiment.txt"
                )
                print(f"Performing sentiment analysis on {file}...")
                sentiment_analysis(input_file_path, output_file_path)
                print(f"Sentiment analysis complete for {file}. Results saved in {os.path.basename(output_file_path)}.\n")