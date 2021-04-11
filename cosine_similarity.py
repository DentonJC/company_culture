#!/usr/bin/env python

import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import process_text


df = pd.read_csv("data/culture500_data.csv", encoding="ISO-8859-1")
df = df.set_index(["company", "value"])["sentiment_score"].unstack()

# df.to_csv("data/culture500_data_preprocessed.csv")

companies_names = list(df.index.values)

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
idx = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)[:10]
print('Cultural match:')
for i, company in enumerate(np.array(companies_names)[idx]): 
    print(str(i+1) + '. ' + company)
