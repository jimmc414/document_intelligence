import os
import re

# Define the input and output folders
input_folder = "c:\\python\\autoindex\\txt_output"
output_folder = "c:\\python\\autoindex\\kvextract"

# Define the regex pattern for key-value pairs
pattern = r"(Our File #|Your File #)\s*:?\s*((?:\d+-\d+-?\d*)|(?:\d{16})|(?:(?:\d{3}-\d{2}-\d{4})))"

# Loop through the txt files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is a txt file
    if filename.endswith(".txt"):
        # Open the txt file and read its content using utf-8 encoding
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as input_file:
            content = input_file.read()

        # Find all the key-value pairs in the text using regex
        kv_pairs = re.findall(pattern, content)
        # Create a new txt file in the output folder with the same name as the input file
        with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as output_file:
            # Write the key-value pairs to the output file, one per line
            for key, value in kv_pairs:
                output_file.write(f"{key}: {value}\n")