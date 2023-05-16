If you want to train your NER model on a list of known names in the system, you would need to do the following steps:

Create a list of text documents that contain the known names you want to recognize. You can use any source of text that is relevant to your domain and language, such as web pages, news articles, books, etc. You can also use synthetic or generated text if you don't have enough real data.

Annotate the text documents with the entity labels. You can use spaCy's built-in annotation tool Prodigy or other tools like Doccano or Label Studio to mark the spans of text that correspond to the entities and assign them a label. For example, you can mark "John Smith" as a PERSON entity and "New York" as a LOCATION entity. You can also use keyboard shortcuts, filters, and other features to speed up the annotation process. You need to follow spaCy's training data format for named entity recognition (NER), which is a list of tuples containing the text and a dictionary of annotations. For example:

[("John Smith lives in New York.", {"entities": [(0, 10, "PERSON"), (19, 27, "LOCATION")]})]

Split your data into training and evaluation sets. You can use a random or stratified split to divide your data into two parts: one for training the model and one for testing its performance. You can use any ratio you want, but a common one is 80% for training and 20% for evaluation.

Convert your data into spaCy's binary format (.spacy) using spacy convert command. This will make your data more efficient and compact for training. You can specify the input and output paths, the language and the converter type (ner for named entity recognition). For example:

python -m spacy convert -t ner -l en -c ner ./train.json ./train.spacy

python -m spacy convert -t ner -l en -c ner ./eval.json ./eval.spacy

Train your NER model using spacy train command with your data and config file. You can use an existing spaCy model as a base model or start from scratch with a blank model. You can also customize your config file with various settings and hyperparameters for your pipeline. For example:

python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./eval.spacy

Hopefully this helps you understand how to train your NER model on a list of known names in the system.