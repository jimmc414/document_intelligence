import requests
import os
from sklearn.extract_features_from_text.text import TfidfVectorizer
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
  # Iterate through the txt files in C:\python\autoindex\txt_output.
  for filename in os.listdir("C:\\python\\autoindex\\txt_output"):
    # Read the file.
    with open(f"C:\\python\\autoindex\\txt_output\\{filename}", "r") as f:
      document = f.read()

    # Classify the document.
    prediction = get_document_classification(document)

    # Write the classification to a file.
    with open(f"C:\\python\\autoindex\\document_classification\\{filename}", "w") as f:
      f.write(prediction)

if __name__ == "__main__":
  main()
