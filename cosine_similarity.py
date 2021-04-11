import os
import pickle
import time

import numpy as np
import pandas as pd
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import process_text


def get_description(query):
    res = wikipedia.search(query + " Company")
    try:
        if "ADP" in res[0]:
            res[0] = "ADP"  # some wikipedia library bug
        page = wikipedia.page(res[0])
    except wikipedia.DisambiguationError as err:
        page = wikipedia.page(err.options[0])
    except wikipedia.exceptions.PageError:
        return "error"
    time.sleep(1)
    return str(page.content.encode("utf-8"))


df = pd.read_csv("data/culture500_data.csv", encoding="ISO-8859-1")
df = df.set_index(["company", "value"])["sentiment_score"].unstack()
companies_names = list(df.index.values)

if not os.path.isfile("data/companies.txt"):
    companies = []
    with open("data/companies.csv", "w") as f:
        f.write("name;content\n")
    for c in tqdm(companies_names):
        try:
            content = get_description(c)
        except:
            time.sleep(10)
            try:
                content = get_description(c)
            except:
                content = "error"
        companies.append(content)
        with open("data/companies.csv", "a") as f:
            f.write(c + ";" + content.replace(";", " ") + "\n")

    with open("data/companies.txt", "wb") as fh:
        pickle.dump(companies, fh)
else:
    infile = open("data/companies.txt", "rb")
    companies = pickle.load(infile)
    infile.close()

ngram_range = (1, 1)
max_features = 500
vect = TfidfVectorizer(
    tokenizer=process_text,
    analyzer="word",
    stop_words="english",
    use_idf=True,
    smooth_idf=True,
    ngram_range=ngram_range,
    max_features=max_features,
)
vect.fit(companies)

letter_file = open("data/letter.txt", "r")
letter = letter_file.read().replace("\n", " ")
letter_file.close()

similarity = cosine_similarity(
    vect.transform(companies), vect.transform([letter])
).flatten()
idx = sorted(range(len(similarity)), key=lambda i: similarity[i])[-10:]
print(np.array(similarity)[idx])
print(np.array(companies_names)[idx])
