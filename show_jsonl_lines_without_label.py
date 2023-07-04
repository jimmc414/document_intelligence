import json

# Get filename
filename = input("Enter the filename of the JSONL file: ")

# Open file
with open(filename, 'r') as file:
    lines = file.readlines()  # read all lines

# Iterate over lines and check if "label" exists and is not empty
for i, line in enumerate(lines):
    data = json.loads(line)
    if "label" not in data or not data["label"]:
        print(f"Line {i+1} does not have a label!")