import os
from transformers import AutoTokenizer, AutoModel

input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex"

# Load the pre-trained tokenizer and RoBERTa model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

def extract_keywords(text):
    tokens = tokenizer(text, return_tensors="pt")
    predictions = model(**tokens).last_hidden_state

    # Apply max pooling over the sequence dimension
    keywords = torch.max(predictions[0], dim=0)[1]

    # Convert the indices to the corresponding tokens
    keyword_tokens = [tokenizer.convert_ids_to_tokens(key_id.item()) for key_id in keywords]

    return keyword_tokens

def process_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(input_dir, file)

            # Read the contents of the file
            with open(file_path, "r", encoding="utf-8") as input_file:
                text = input_file.read()

            # Perform keyword extraction
            results = extract_keywords(text)

            output_file_name = f"NER_Extraction_{file}"
            output_file_path = os.path.join(output_dir, output_file_name)

            # Save keywords to output file
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for keyword in results:
                    output_file.write(f"{keyword}\n")

os.makedirs(output_dir, exist_ok=True)
process_files(input_dir, output_dir)