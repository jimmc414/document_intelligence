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
  document1_path = input("Enter the path and filename of the file to compare to: ")

  # Read the content of document1
  with open(document1_path, 'r', encoding='utf-8') as f:
    document1_content = f.read()

  # Define paths
  txt_output_dir = os.path.join(os.getcwd(), "txt_output")

  # Store all similarity scores for ranking
  all_scores = []

  # Iterate through the txt files
  for filename in os.listdir(txt_output_dir):
    if not filename.endswith(".txt"):
      continue

    file_path = os.path.join(txt_output_dir, filename)

    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as f:
      document2_content = f.read()

    # Calculate the similarity score.
    similarity_score = get_document_similarity(document1_content, document2_content)

    # Store the result
    all_scores.append((filename, similarity_score[0][0]))

  # Sort by similarity score in descending order
  all_scores.sort(key=lambda x: x[1], reverse=True)

  # Print the similarity scores
  print("\nSimilarity scores (ranked):")
  for filename, score in all_scores:
    print(f"{filename}: {score:.4f}")

if __name__ == "__main__":
  main()
