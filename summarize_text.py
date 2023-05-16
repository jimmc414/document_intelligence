import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def create_summary(text, language="english", sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)

    # Use LsaSummarizer, but you can also try other summarizers like LexRankSummarizer or LuhnSummarizer
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    
    summary_sentences = summarizer(parser.document, sentence_count)
    
    summary = " ".join([str(sentence) for sentence in summary_sentences])
    return summary


input_dir = "c:\\python\\autoindex\\txt_output"
output_dir = "c:\\python\\autoindex\\summarization"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_dir, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Generate summary
        summary = create_summary(text)

        # Save the summary in the summarization folder
        output_filename = "summary_" + filename
        output_filepath = os.path.join(output_dir, output_filename)

        with open(output_filepath, "w", encoding="utf-8") as output_file:
            output_file.write(summary)