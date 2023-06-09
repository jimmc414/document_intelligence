The fuzzy_match_text.py script in the document_intelligence project uses the fuzz library to perform fuzzy matching on text. The script takes two arguments: a list of entities and a dictionary of accounts. The entities are the names or addresses that you want to match, and the accounts dictionary contains a list of names and addresses for each account.

The script works by iterating over the entities and using the fuzz.extractOne() function to find the best match in the accounts dictionary. The fuzz.extractOne() function takes three arguments: the entity, the choices, and the scorer. The entity is the name or address that you want to match, the choices are the list of names or addresses in the accounts dictionary, and the scorer is a function that calculates the similarity between two strings.

The fuzz.extractOne() function returns a tuple containing the best match and its similarity score. The script then adds the match to a list of matches. After iterating over all of the entities, the script writes the list of matches to a CSV file.

Here is an example of how to use the fuzzy_match_text.py script:

Python

entities = ["John Smith", "123 Main Street"]

accounts = {

    "John Smith": ["John Smith", "123 Main Street"],

    "Jane Doe": ["Jane Doe", "456 Elm Street"],

}

fuzzy_match_text.py(entities, accounts)

This will write the following CSV file to the current directory:

Code snippet

doc_name,confidence_score

John Smith,100

123 Main Street,100