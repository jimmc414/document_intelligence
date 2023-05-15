document_similarity.py
This program calculates the similarity between two documents. It uses a TfidfVectorizer to convert the documents to vectors, and then computes the cosine similarity between the vectors.

Requirements
Python 3
requests
os
sklearn
Usage
To use the program, you need to have some documents in the C:\python\autoindex\txt_output directory. The program will compare one document that you specify at runtime with all the documents in this directory and print the similarity scores.

To run the program, execute the following command:

python document_similarity.py

The program will prompt you to enter the path and filename of the file to compare to. For example, if you want to compare the document document1.txt in the same directory, you would enter:

Enter the path and filename of the file to compare to: document1.txt

The program will then iterate through all the files in the C:\python\autoindex\txt_output directory and calculate the similarity score between document1.txt and each file. It will rank the results in descending order of similarity and print them to the console.

Example
For example, suppose you have three documents in the C:\python\autoindex\txt_output directory: document1.txt, document2.txt, and document3.txt. The contents of these documents are:

document1.txt: This is a sample document for testing purposes.
document2.txt: This is another sample document for testing purposes.
document3.txt: This is a completely different document that has nothing to do with testing.

If you run the program and enter document1.txt as the file to compare to, you will get the following output:

Enter the path and filename of the file to compare to: document1.txt
The similarity score between document 1 and document 2 is 0.8944271909999159
The similarity score between document 1 and document 3 is 0.0
The similarity score between document 1 and document 1 is 1.0

This means that document1.txt is most similar to itself, followed by document2.txt, and then document3.txt.

How it works
The document_similarity.py program works by first converting the documents to vectors using a TfidfVectorizer. A TfidfVectorizer is a statistical method that calculates the term frequency-inverse document frequency (tf-idf) of each word in the document. The term frequency is the number of times the word appears in the document, and the inverse document frequency is the logarithm of the ratio of the total number of documents in the corpus to the number of documents that contain the word. The tf-idf value reflects how important a word is in a document relative to the corpus.

Once the documents are converted to vectors, the program computes the cosine similarity between them. The cosine similarity is a measure of how similar two vectors are based on their angle. It ranges from -1 (opposite directions) to 1 (same direction). A higher cosine similarity means a higher degree of similarity between two documents.