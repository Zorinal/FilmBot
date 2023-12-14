import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
MAX_SENTENCE_LEN = 512
device = torch.device('cuda')
class MyLSTM(torch.nn.Module):
    def __init__(self,dimension, hidden_size, num_layers,out_size,bidirectional):
        super(MyLSTM,self).__init__()
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(dimension, hidden_size, num_layers,batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, out_size,dtype=torch.float32)

    def forward(self, sentence):
        out, _ = self.lstm(sentence.view(len(sentence),1,-1))
        tag_space = self.fc(out.view(len(sentence),-1))
        tag_score = torch.nn.functional.log_softmax(tag_space,dim = 1)
        return tag_score


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
            temp_val = dict_vocab[temp_sent[word].lower()]
            np.put(temp_tokenize_sentences, (0, word), temp_val)
        tokenize_sentences = np.vstack((tokenize_sentences, temp_tokenize_sentences))
    tokenize_sentences = np.delete(tokenize_sentences, 0, axis=0)
    return tokenize_sentences


def create_vocab_and_sentences_and_answers(input_sentences):
    vocab = []
    sentences = []
    answers = []
    for index in range(len(input_sentences)):
        try:
            vocab.extend(list(map(lambda x: x, films["description"][index].split())))
            vocab.extend(list(map(lambda x: x, films["title" or "movie_name"][index].split())))
            sentences.append(films["description"][index])
            answers.append(films['title'][index])
        except:
            continue

    vocab = list(set(vocab))
    dict_vocab = {}

    for i in range(len(vocab)):
        dict_vocab.update({vocab[i].lower(): i + 1})

    return dict_vocab, sentences, answers


films = pd.read_csv("../datasets/filmtv_movies.csv")
# tokenize sentences by words
dict_vocab = create_vocab_and_sentences_and_answers(films[:2000])[0]
sentences = create_vocab_and_sentences_and_answers(films[:2000])[1]
answers = create_vocab_and_sentences_and_answers(films[:2000])[2]


tokenize_sentences = tokenize(sentences,dict_vocab)
tokenize_answers = tokenize(answers,dict_vocab)

for i in range(2000):
    tokenize_sentences[i] = semantic_sense(tokenize_sentences[i],tokenize_sentences)
    tokenize_sentences[i] = soft_max(tokenize_sentences[i])
    tokenize_answers[i] = soft_max(tokenize_answers[i])

dataset = TensorDataset(torch.tensor(tokenize_sentences, dtype=torch.float32),torch.tensor(tokenize_answers, dtype=torch.float32))
train_size = int(0.9*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)
dimension = 512
hide_size = 512
num_layers = 10
out_size = 512
bidirectional = True
model = MyLSTM(dimension,hide_size,num_layers,out_size,bidirectional)
model = model.to(device)
num_epoch = 30
loss = torch.nn.CrossEntropyLoss() #CEloss = -sum(target[i] * log(prediction[i]) or BCELoss = -1/N * sum(target[i] * log(prediction[i] + (1-target[i])log(1-prediction[i]))
optimizer = torch.optim.Adam(model.parameters())
#print(model)
#train process
for epoch in range(num_epoch):
    for description,answer in train_loader:
        optimizer.zero_grad()
        y_prediction = model(description)
        _loss = loss(y_prediction, answer)
        _loss.backward()
        optimizer.step()
#train = false
#model.eval()
#evulation results
#with torch.zero_grad():
   # for description, answer in train_data[1601:1999]:
    #    y_prediction = model(description)
    #    print(y_prediction)

#input_description= soft_max(tokenize(str(input()), dict_vocab))
#prediction = model(input_description)
#print(prediction)