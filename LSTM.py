import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class MyLSTM(torch.nn.Module):
    def __init__(self,dimension, hidden_size, num_layers,out_size):
        super(MyLSTM,self).__init__()
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.batch_size = self.num_layer
        self.lstm = torch.nn.LSTM(self.dimension, self.hidden_size, self.num_layer)

        self.fc = torch.nn.Linear(hidden_size, out_size,dtype=torch.float32)

    def forward(self, sentence):
        #h0 = torch.zeros(self.num_layer,sentence.size(0), self.hidden_size*self.num_layer,dtype=torch.float32)
        out,(h0,c0)= self.lstm(sentence)
        out = self.fc(h0)
        out = torch.nn.functional.softmax(out,dim=1)
        return out


def tokenize(input_sentences, dict_vocab):
    MAX_SENTENCE_LEN = len(dict_vocab)
    out_tokenize_sentences = np.zeros([MAX_SENTENCE_LEN])
    for i in range(len(input_sentences)):
        temp_sent = input_sentences[i].split()
        temp_tokenize_sentences = np.zeros([MAX_SENTENCE_LEN])
        for word_ind in range(len(temp_sent)):
            key = [k for k,v in dict_vocab.items() if v == temp_sent[word_ind]]
            np.put(temp_tokenize_sentences, (0, word_ind), key)
        out_tokenize_sentences = np.vstack((out_tokenize_sentences, temp_tokenize_sentences))
    out_tokenize_sentences = np.delete(out_tokenize_sentences, 0, axis=0)
    return out_tokenize_sentences


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
        dict_vocab.update({i+1:vocab[i].lower()})

    return dict_vocab, sentences, answers


films = pd.read_csv("../datasets/filmtv_movies.csv")
# tokenize sentences by words
dict_vocab = create_vocab_and_sentences_and_answers(films[:1100])[0]
sentences = create_vocab_and_sentences_and_answers(films[:1100])[1]
answers = create_vocab_and_sentences_and_answers(films[:1100])[2]


tokenize_sentences = tokenize(sentences,dict_vocab)
tokenize_answers = tokenize(answers,dict_vocab)

dataset = TensorDataset(torch.tensor(tokenize_sentences, dtype=torch.float32),torch.tensor(tokenize_answers, dtype=torch.float32))
train_size = int(0.9*len(dataset))
test_size = len(dataset) - train_size
hide_size = 1100
num_layers = 30
out_size = len(dict_vocab)
train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_dataset,batch_size=num_layers)
test_loader = DataLoader(test_dataset)
dimension = len(dict_vocab)

#model = MyLSTM(dimension,hide_size,num_layers,out_size)
model = torch.jit.load('model_scripted1100softmax(x2layers,epochs+5).pt')
model.eval()
#model = model.to(device)
num_epoch = 20
#model.eval()
loss = torch.nn.CrossEntropyLoss() #CEloss = -sum(target[i] * log(prediction[i]) or BCELoss = -1/N * sum(target[i] * log(prediction[i] + (1-target[i])log(1-prediction[i]))
optimizer = torch.optim.Adam(model.parameters())
print(model)
#train process
#for epoch in range(num_epoch):
 # for description,answer in train_loader:
      #  optimizer.zero_grad()
      #  y_prediction = model(description)
       # _loss = loss(y_prediction, answer)
        #_loss.backward()
      #  optimizer.step()
#model_scripted = torch.jit.script(model) # Export to TorchScript
#model_scripted.save('model_scripted1100softmax(x2layers,epochs+5).pt') # Save
#train = false
#model.eval()
#evulation results
#with torch.no_grad():
   # for description, answer in test_loader:
    #    y_prediction = model(description)
     #   print(y_prediction)
input_description = tokenize(str(input()), dict_vocab)
input_description = torch.tensor(input_description,dtype=torch.float32)
word = []
prediction = model(input_description)
print(prediction)
for i in range(len(prediction)):
   word.append(torch.argmax(prediction[i]).item())
#word = list(set(word))
for i in range(len(word)):
    tmp = i+1
    print(dict_vocab[tmp],end = ' ')