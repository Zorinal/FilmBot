import encoder
import gpt2
import utils
from models import *

import numpy as np
import pandas as pd
import ast

data = pd.read_csv("movies_metadata.csv", low_memory=False)
data = data[['adult', 'genres', 'original_language', 'title', 'overview', 'release_date', 'production_companies',
             'production_countries', 'vote_average', 'vote_count', 'runtime']]
data['original_language'] = pd.factorize(data['original_language'])[0]
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce').dt.year
data = data.dropna()


def extract_ids_from_keywords_list(df_column):
    words_list = []
    for keywords_str in df_column:
        keywords_list = ast.literal_eval(keywords_str)  # Преобразование строки в список словарей
        words = [keyword['id'] for keyword in keywords_list]
        words_list.append(words)
    return words_list


def extract_words_from_keywords_list(df_column):
    words_list = []
    for keywords_str in df_column:
        keywords_list = ast.literal_eval(keywords_str)  # Преобразование строки в список словарей
        words = [keyword['name'] for keyword in keywords_list]
        words_list.append(words)
    return words_list


data['genres'] = extract_ids_from_keywords_list(data['genres'])
data['production_companies'] = extract_ids_from_keywords_list(data['production_companies'])
data['production_countries'] = extract_words_from_keywords_list(data['production_countries'])


def count_matching_words(string1, string2):
    words1 = string1.split()
    words2 = string2.split()

    matching_words = {}

    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                if word1 in matching_words:
                    matching_words[word1] += 1
                else:
                    matching_words[word1] = 1
    return len(matching_words)


#print(data.at[1, 'overview'])
words = gpt2.main(input())
max_matchings = 0
best_film = ''
print(data.shape[0])
n = 0
for i in range(data.shape[0]):
    try:
        if max_matchings < count_matching_words(words, data.at[i, 'overview']):
            best_film = data.at[i, 'title']
            max_matchings = count_matching_words(words, data.at[i, 'overview'])
            n += 1
    except:
        ...
print(best_film)
# print(gpt2.main("Alan Turing theorized that computers would one day become"))
