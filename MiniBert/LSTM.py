import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import keras
device = torch.device('cuda')
MAX_SENTENCE_LEN = 512
class MyLSTM(torch.nn.Module):
    def __init__(self,dimension, hidden_size, num_layers,out_size,bidirectional):
        super(MyLSTM,self).__init__()
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(dimension, hidden_size, num_layers, bidirectional= bidirectional )

        self.fc = torch.nn.Linear(dimension, out_size)

    def forward(self, inp):
        h0 = torch.zeros(self.num_layer,inp.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer,inp.size(0),self.hidden_size).to(device)

        out, _ = self.lstm(inp, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out




def semantic_sense(first_vector,all_vector):
    for i in range(len(all_vector)):
        try:
            scalar_mul = first_vector*all_vector[i]
            first_vector[i] = scalar_mul
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




imdb = pd.read_csv("../datasets/imdb.csv")
films = pd.read_csv("../datasets/filmtv_movies.csv")
# tokenize sentences by words
vocab = []
sentences = []
answers = []
for i in range(1500):
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

for i in range(1500):
    tokenize_sentences[i] = semantic_sense(tokenize_sentences[i],tokenize_sentences)
    tokenize_sentences[i] = soft_max(tokenize_sentences[i])
    tokenize_answers[i] = soft_max(tokenize_answers[i])
X_train, X_test, y_train, y_test = train_test_split(tokenize_sentences, tokenize_answers, test_size=0.2)
criterion = torch.nn.MSELoss()
dimension = 512
hide_size = 5
num_layers = 512
out_size = 10
bidirectional = True
model = MyLSTM(dimension,hide_size,num_layers,out_size,bidirectional)
num_epoch = 100
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epoch):
    for X,y in zip(tokenize_sentences, tokenize_answers):
        optimizer.zero_grad()
        y_prediction = model(X.unsqueeze(-1))
        loss = loss(y_prediction.squeeze(),y)
        loss.backward()
        optimizer.step()