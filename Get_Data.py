import torch
import pandas as pd
import numpy as np
import matplotlib as mlt
from sklearn import svm
import matplotlib.pyplot as plt

MAX_SENTENCE_LEN = 512
def semantic_sense(first_vector,all_vector):
    for i in range(len(all_vector)):
        try:
            scalar = first_vector*all_vector[i]
            first_vector[i] = scalar
        except:
            continue
    return first_vector


def soft_max(input_vector: np.array):
    sum_vector = sum(input_vector)
    for i in range(len(input_vector)):
        if sum_vector != 0:
            input_vector[i] = input_vector[i] / sum_vector
    return input_vector

def tokenize(input_sentences, dict_vocab):
    tokenize_sentences = np.zeros([MAX_SENTENCE_LEN])
    for i in range(len(input_sentences)):
        temp_sent = input_sentences[i].split()
        temp_tokenize_sentences = np.zeros([MAX_SENTENCE_LEN])
        for word in range(len(temp_sent)):
            temp_val = dict_vocab[temp_sent[word]]
            np.put(temp_tokenize_sentences, (0, word), temp_val)
        tokenize_sentences = np.vstack((tokenize_sentences, temp_tokenize_sentences))
    tokenize_sentences = np.delete(tokenize_sentences, 0, axis=0)
    return tokenize_sentences

def main():
    imdb = pd.read_csv("../datasets/imdb.csv")
    films = pd.read_csv("../datasets/filmtv_movies.csv")
    # tokenize sentences by words
    vocab = []
    sentences = []
    answers = []
    for i in range(1000):
        try:
            vocab.extend(list(map(lambda x: x, films["description"][i].split())))
            vocab.extend(list(map(lambda x: x, films["title"][i].split())))
            sentences.append(films["description"][i])
            answers.append(films['title'][i])
        except:
            continue
    vocab = list(set(vocab))

    dict_vocab = {}

    for i in range(len(vocab)):
        dict_vocab.update({vocab[i]: i + 1})
    tokenize_sentences = tokenize(sentences,dict_vocab)
    tokenize_answers = tokenize(answers,dict_vocab)

    for i in range(1000):
        tokenize_sentences[i] = semantic_sense(tokenize_sentences[i],tokenize_sentences)
        tokenize_sentences[i] = soft_max(tokenize_sentences[i])
        tokenize_answers[i] = soft_max(tokenize_answers[i])



if __name__ == '__main__':
    main()
