import torch
import pandas as pd
import numpy as np
import matplotlib as mlt
from sklearn import svm

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


def main():
    imdb = pd.read_csv("../datasets/imdb.csv")
    films = pd.read_csv("../datasets/filmtv_movies.csv")
    # tokenize sentences by words
    vocab = []
    sentences = []
    answers = []
    for i in range(len(films)):
        try:
            vocab.extend(list(map(lambda x: x, films["description"][i].split())))
            vocab.extend(list(map(lambda x: x, films["title"][i].split())))
            sentences.append(films["description"][i])
            answers.append(films['title'][i])
        except:
            continue
    vocab = list(set(vocab))

    dict = {}
    max_len_sentences = len(max(sentences, key=len))
    max_len_answer = len(max(answers, key=len))
    for i in range(len(vocab)):
        dict.update({vocab[i]: i + 1})
    tokenize_sentences = np.zeros([max_len_sentences])
    tokenize_answers = np.zeros([max_len_answer])

    for i in range(100):
        temp_sent = sentences[i].split()
        temp_answ = answers[i].split()
        temp_tokenize_sentences = np.zeros([max_len_sentences])
        temp_tokenize_answer = np.zeros([max_len_answer])
        for word in range(len(temp_sent)):
            temp_val = dict[temp_sent[word]]
            np.put(temp_tokenize_sentences, (0, word), temp_val)
        for ind_ans in range(len(temp_answ)):
            temp_val = dict[temp_answ[ind_ans]]
            np.put(temp_tokenize_answer,(0,ind_ans),temp_val )
        tokenize_sentences = np.vstack((tokenize_sentences, temp_tokenize_sentences))
        tokenize_answers = np.vstack((tokenize_answers, temp_tokenize_answer))
    for i in range(100):
        tokenize_sentences[i] = semantic_sense(tokenize_sentences[i],tokenize_sentences)
        tokenize_sentences[i] = soft_max(tokenize_sentences[i])
        tokenize_answers[i] = soft_max(tokenize_answers[i])

    print(tokenize_answers[3])



if __name__ == '__main__':
    main()
