import re
import string

from nltk import word_tokenize
from nltk.stem import PorterStemmer


def process_text(text, stem=True):
    """ Clean and tokenize text, stem words removing punctuation """
    stops = [line.strip() for line in open("data/stop_words_english.txt", "r")]

    # Remove all the special characters
    text = re.sub(r"\W", " ", text)

    # remove all single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)

    # Remove single characters from the start
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)

    # Substituting multiple spaces with single space
    text = re.sub(r"\s+", " ", text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r"^b\s+", "", text)

    text = text.lower()
    text = text.translate(string.punctuation)
    toks = word_tokenize(text)

    stops = [s.replace("'", "") for s in stops]

    tokens = []
    for token in toks:
        if token.isalpha() and token not in stops:
            tokens.append(token)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if len(stemmer.stem(t)) > 3]
    return tokens
