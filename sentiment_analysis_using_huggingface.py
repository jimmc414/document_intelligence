# This code implements a sentiment analysis function to identify emotions, such as joy, anger, fear, and sadness, using the Hugging Face Transformers library.

# 1. Import the necessary libraries (`transformers` and `torch`).
# 2. Load the pre-trained tokenizer and model using `AutoTokenizer` and `AutoModelForSequenceClassification`.
# 3. Define a `sentiment_analysis()` function that takes text input.
# 4. Tokenize the input text using the pre-trained tokenizer to convert it into a format the model can understand (a list of token IDs). Additionally, set `return_tensors='pt'` to get results as a PyTorch tensor. Use truncation and padding to ensure consistent input lengths.
# 5. Run the model by passing the tokenized input. This step returns the logits (raw output) for each emotion from the pre-trained model.
# 6. Apply the softmax function to the logits to convert them into probabilities.
# 7. Retrieve the emotion labels from the tokenizer.
# 8. Map the emotion labels to their corresponding probabilities using a zip function.
# 9. Sort the results by probability in descending order, so the most likely emotion is listed first.
# 10. Test the sentiment_analysis function with a sample text and print the sorted results, displaying the probabilities of each emotion.

# Import required libraries
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

# Manually define emotion labels
emotions = ['anger', 'joy', 'optimism', 'sadness']

# Function to perform sentiment analysis
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    results = dict(zip(emotions, probabilities[0].tolist()))
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results

# Function to process all TXT files in the input directory
def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)
            with open(file_path, "r", encoding="utf-8") as input_file:
                text = input_file.read()

            results = sentiment_analysis(text)

            output_file_path = os.path.join(output_dir, f"HF_{file}")

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for emotion, probability in results.items():
                    output_file.write(f"{emotion}: {probability}\n")

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\HF_sentiment"
os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)


# FINETUNING

# The following steps will guide you through the process of fine-tuning a pre-trained BERT model using Hugging Face Transformers and your domain-specific labeled data with emotion labels for sentiment analysis:

# 1. Prepare your dataset: Collect and create a labeled dataset for sentiment analysis based on your domain-specific data. Your dataset should have a consistent format, where each instance has a text and a corresponding emotion label. The dataset should be split into training, validation, and testing sets. Common formats for dataset files include CSV, TSV, or JSON.

# 2. Install necessary libraries: Make sure you have the Hugging Face Transformers and torch libraries installed in your environment. If not, you can install them using:

# ```bash
# pip install transformers torch
# ```

# 3. Create a Python script or a Jupyter notebook for your fine-tuning process.

# 4. Load your dataset: Use the Pandas library to load your train and validation datasets into DataFrames.

# ```python
# import pandas as pd

# train_df = pd.read_csv("<path_to_training_set>.csv")
# val_df = pd.read_csv("<path_to_validation_set>.csv")
# ```

# 5. Preprocess the dataset: If needed, preprocess your text data to remove unwanted elements, such as HTML tags or unusual characters.

# 6. Encode the dataset: Tokenize and encode your dataset using the pre-trained tokenizer. Adjust the `max_length` value based on the expected length of your text data.

# ```python
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# def encode_data(df):
    # return tokenizer.batch_encode_plus(
        # df['text'].tolist(),
        # padding='max_length',
        # truncation=True,
        # max_length=256,
        # return_tensors="pt"
    # )

# train_encodings = encode_data(train_df)
# val_encodings = encode_data(val_df)
# ```

# 7. Convert the emotion labels into integers for classification:

# ```python
# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()
# train_labels = label_encoder.fit_transform(train_df['emotion'])
# val_labels = label_encoder.transform(val_df['emotion'])
# ```

# 8. Create a custom PyTorch Dataset:

# ```python
# import torch

# class EmotionDataset(torch.utils.data.Dataset):
    # def __init__(self, encodings, labels):
        # self.encodings = encodings
        # self.labels = labels

    # def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        # return item

    # def __len__(self):
        # return len(self.labels)

# train_dataset = EmotionDataset(train_encodings, train_labels)
# val_dataset = EmotionDataset(val_encodings, val_labels)
# ```

# 9. Load the pre-trained BERT model:

# ```python
# from transformers import DistilBertForSequenceClassification

# NUM_LABELS = len(label_encoder.classes_)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS)
# ```

# 10. Set up the training arguments:

# ```python
# from transformers import TrainingArguments

# training_args = TrainingArguments(
    # output_dir='./results',
    # num_train_epochs=3,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=64,
    # warmup_steps=500,
    # weight_decay=0.01,
    # logging_dir='./logs',
    # logging_steps=10
# )
# ```

# 11. Set up the Trainer and fine-tune the model:

# ```python
# from transformers import Trainer

# trainer = Trainer(
    # model=model,
    # args=training_args,
    # tokenizer=tokenizer,
    # train_dataset=train_dataset,
    # eval_dataset=val_dataset
# )

# trainer.train()
# ```

# 12. Save the fine-tuned model:

# ```python
# trainer.save_model("./fine_tuned_emotion_model")
# ```

# Now that you have fine-tuned the BERT model using your domain-specific labeled data, you can use it for sentiment analysis of your text data following a similar approach as in the previous example implementation, but loading the tokenizer and model from the local fine-tuned model directory:

# ```python
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_emotion_model")
# model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_emotion_model")
# ```

# Remember to adjust the specific tokenizer, model names, and dataset paths based on your environment and requirements.

# For more detailed instructions and examples, please refer to the [official Hugging Face documentation](https://huggingface.co/transformers/training.html).

# When fine-tuning or training a model using Hugging Face Transformers or similar libraries, the input data format should typically include two main components: the text data and corresponding labels. Here's a general guideline for preparing your data:

# 1. Text data: The text data should be in a raw or plain text format, which can be a list of sentences, paragraphs, or documents, depending on your task requirements.

# 2. Labels: The labels should correspond to each text entry. Label formats can be integers or strings; however, it is common to use integer-encoded labels for training machine learning models.

# A common way to store this data is in tabular formats such as CSV, TSV, or JSON. For CSV or TSV formats, each row should contain a text entry and its associated label, separated by a delimiter (comma for CSV, tab for TSV). For JSON, you can represent the data as a list of dictionaries, where each dictionary contains the text and the associated label.

# Here's an example of data representation in different formats:

# CSV format:
# ```
# text,label
# "Sample sentence 1",0
# "Sample sentence 2",1
# "Sample sentence 3",0
# ```

# TSV format:
# ```
# text	label
# Sample sentence 1	0
# Sample sentence 2	1
# Sample sentence 3	0
# ```

# JSON format:
# ```json
# [
  # {"text": "Sample sentence 1", "label": 0},
  # {"text": "Sample sentence 2", "label": 1},
  # {"text": "Sample sentence 3", "label": 0}
# ]
# ```

# Before using the data for fine-tuning or training, you should preprocess and convert the text data into a format compatible with the specific tokenizer and model you are using. For Hugging Face Transformers, this typically involves tokenizing the text data into a list of token IDs and organizing input data as PyTorch tensors, TensorFlow tensors, or NumPy arrays, depending on your library of choice.