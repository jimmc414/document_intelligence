import json

def remove_first_line():
    jsonl_file = input("Please enter the input JSONL file path: ")
    output_file = input("Please enter the output JSONL file path: ")

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    for item in data:
        text = item['text']
        first_line, rest_text = text.split("\n", 1)
        # if the first line ends with "_ocr" remove it
        if first_line.endswith('_ocr'):
            item['text'] = rest_text
        # if the first line does not end with "_ocr", keep it
        else:
            item['text'] = text

    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

    print(f'Done. The processed data has been saved to {output_file}')

remove_first_line()