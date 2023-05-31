import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

def get_document_similarity(document1, document2):
  """
  Calculates the similarity between two documents.

  Args:
    document1: The first document.
    document2: The second document.

  Returns:
    The similarity score between the two documents.
  """

  # Convert the documents to vectors.
  vectorizer.fit([document1, document2])
  document1_vector = vectorizer.transform([document1])
  document2_vector = vectorizer.transform([document2])

  # Calculate the similarity score.
  import numpy as np

  document1_vector = document1_vector.reshape(1, -1)
  document2_vector = document2_vector.reshape(1, -1)

  similarity_score = cosine_similarity(document1_vector, document2_vector)

  return similarity_score


def main():
  # Define document 1 at runtime.
  document1 = input("Enter the path and filename of the file to compare to: ")

  # Iterate through the txt files in C:\python\autoindex\txt_output.
  for filename in os.listdir("C:\\python\\autoindex\\txt_output"):
    # Calculate the similarity score.
    similarity_score = get_document_similarity(document1, filename)

    # Rank the results in descending order of similarity.
    similarity_scores = [similarity_score]
    similarity_scores.sort(reverse=True)

    # Print the similarity score.
    print(f"The similarity score between document 1 and document {filename} is {similarity_score}")

if __name__ == "__main__":
  main()
