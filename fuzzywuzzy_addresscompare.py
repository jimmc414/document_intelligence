import requests
from fuzzywuzzy import fuzz

def get_address_similarity(address1, address2):
  """
  Calculates the similarity score for a pair of addresses.

  Args:
    address1: The first address.
    address2: The second address.

  Returns:
    The similarity score.
  """

  # Get the similarity score using fuzzywuzzy.
  similarity_score = fuzz.ratio(address1, address2)

  # Return the similarity score.
  return similarity_score

if __name__ == "__main__":
  address1 = "1600 Pennsylvania Avenue NW, Washington, D.C. 20500"
  address2 = "1600 Pennsylvania Ave. NW, Washington, D.C., 20500"

  similarity_score = get_address_similarity(address1, address2)

  print(similarity_score)
