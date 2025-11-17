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

  # Read document 1 content
  if not os.path.exists(document1_path):
    print(f"Error: File '{document1_path}' does not exist.")
    return

  with open(document1_path, "r", encoding="utf-8") as f:
    document1 = f.read()

  # Get output directory from environment or use default
  txt_output_dir = os.getenv("TXT_OUTPUT_DIR", "txt_output")

  if not os.path.exists(txt_output_dir):
    print(f"Error: Directory '{txt_output_dir}' does not exist.")
    return

  # Store similarity scores for ranking
  similarity_results = []

  # Iterate through the txt files
  for filename in os.listdir(txt_output_dir):
    if not filename.endswith(".txt"):
      continue

    file_path = os.path.join(txt_output_dir, filename)

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
      document2 = f.read()

    # Calculate the similarity score.
    similarity_score = get_document_similarity(document1, document2)

    similarity_results.append((filename, similarity_score[0][0]))

  # Sort results in descending order of similarity.
  similarity_results.sort(key=lambda x: x[1], reverse=True)

  # Print the similarity scores.
  print("\nDocument Similarity Results (sorted by similarity):")
  print("=" * 60)
  for filename, score in similarity_results:
    print(f"{filename}: {score:.4f}")

if __name__ == "__main__":
  main()
