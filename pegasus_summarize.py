from transformers import pipeline
import os

# create a summarizer with PEGASUS model
summarizer = pipeline("summarization", model="google/pegasus-xsum")

# set the input and output directories
input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\Pegasus_summarization"

# create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# set the chunk size
chunk_size = 501

# iterate through the txt files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        # get the input file path
        input_file_path = os.path.join(input_dir, filename)
        # read the input file content
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            text = input_file.read()
        # split the text into chunks of chunk_size tokens
        tokens = summarizer.tokenizer(text, return_tensors="pt").input_ids[0]
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
        # generate a summary for each chunk
        summaries = []
        for chunk in chunks:
            # decode the chunk with the tokenizer
            chunk_text = summarizer.tokenizer.decode(chunk, skip_special_tokens=True)
            # generate a summary for the chunk text
            summary = summarizer(chunk_text)
            summaries.append(summary[0]["summary_text"])
        # join the summaries into one text
        final_summary = "\n".join(summaries)
        # get the output file path
        output_file_path = os.path.join(output_dir, filename)
        # write the final summary to the output file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(final_summary)