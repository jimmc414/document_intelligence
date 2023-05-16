import os
import stanza
from stanza.server import CoreNLPClient # Import the client module
from stanza.server.client import PermanentlyFailedException


# Download the Stanford CoreNLP package and English models
stanza.install_corenlp()
stanza.download_corenlp_models(model='english-kbp', version='4.2.0')

# Create a client object that uses the CoreNLP server and includes the kbp processor
client = CoreNLPClient(annotators="tokenize,pos,lemma,depparse,sentiment,ner,kbp".split(), kbp_model_name='english-kbp', timeout=600000, memory='6G', use_gpu=True, pos_batch_size=16)


# Create the output directory if it doesn't exist
os.makedirs("C:\\python\\autoindex\\relation_extraction\\", exist_ok=True)

# Iterate through the txt files in C:\python\autoindex\txt_output.
for filename in os.listdir("C:\\python\\autoindex\\txt_output"):
  # Read the file.
  with open(f"C:\\python\\autoindex\\txt_output\\{filename}", "r") as f:
    document = f.read()

  # Try to annotate the document with the client
  try:
    doc = client.annotate(document)
  except PermanentlyFailedException as e:
    # If the server cannot start because of port conflict, try to stop the previous server and start a new one
    if "unable to start the CoreNLP server on port" in str(e):
      print("Trying to stop the previous server and start a new one...")
      client.stop()
      client.start()
      doc = client.annotate(document)
    else:
      # If the error is not related to port conflict, raise it
      raise e

  # Extract the relations from the document.
  relations = []
  for sentence in doc.sentence:
    relations.extend(sentence.relation)

  # Split the filename and the extension
  filename, extension = os.path.splitext(filename)

  # Write the relations to a file using the filename without the extension
  with open(f"C:\\python\\autoindex\\relation_extraction\\{filename}_relations.txt", "w") as f:
    for relation in relations:
      f.write(str(relation) + "\n")

# Close the client when done
client.close()
