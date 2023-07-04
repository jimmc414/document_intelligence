import os

# Set the directory you want to start from
dir_path = r"C:\python\autoindex\txt_output"

for filename in os.listdir(dir_path):
    if filename.endswith(".txt"):
        with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as file:
            content = file.readlines()

        with open(os.path.join(dir_path, filename), "w", encoding="utf-8") as file:
            file.write(filename.rstrip('.txt') + '\n')
            file.writelines(content)