import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Get file name from user and load data
training_file = input("Please enter the training JSONL file path: ")
with open(training_file, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

# Split the data into the document texts and labels
texts = [item["text"] for item in data]
labels = [item["label"][0] for item in data]  # adjust if needed

# Convert labels into numerical format
unique_labels = list(set(labels))
labels = [unique_labels.index(label) for label in labels]

# Split data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2
)

# Initialize a tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Create a torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# Define the model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(unique_labels)
)

# Select a training device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,  # Log every X steps
    evaluation_strategy="steps",  # Evaluate every X steps
    save_strategy="steps",  # Save every X steps
    load_best_model_at_end=True,  # Load the best model when finished training (defaults to True)
    push_to_hub=False,  # Do not push the model to the HuggingFace.co hub
)

# Create the Trainer and train
trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
)

# Train the model
trainer.train()

# Once training is completed you can save the model this way
trainer.save_model("trained_model")