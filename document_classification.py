import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def get_document_classification(document):
  """
  Classifies a document into a particular category.

  Args:
    document: The document to be classified.

  Returns:
    The category of the document.
  """
  document_category = "some_category"

  # Convert the document to a vector.
  vectorizer = TfidfVectorizer()
  vectorizer.fit([document])
  document_vector = vectorizer.transform([document])

  # Train a logistic regression model.
  model = LogisticRegression()
  model.fit(vectorizer.fit_transform([document]), [document_category])

  # Classify the document.
  prediction = model.predict(document_vector)

  return prediction

def main():
  # Define paths
  input_dir = os.path.join(os.getcwd(), "txt_output")
  output_dir = os.path.join(os.getcwd(), "document_classification")

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Iterate through the txt files
  for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
      continue

    # Read the file.
    input_path = os.path.join(input_dir, filename)
    with open(input_path, "r") as f:
      document = f.read()

    # Classify the document.
    prediction = get_document_classification(document)

    # Write the classification to a file (convert numpy array to string).
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
      f.write(str(prediction[0]))

if __name__ == "__main__":
  main()
