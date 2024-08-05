import sys
sys.path
sys.path.append(r'C:\Users\dell\surveyApi\pythonProject')
import string
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\dell\PycharmProjectfx\pythonProject\Symptom2Disease.csv")
stop_words = set(stopwords.words('english'))


def clean_text(sent):
    # remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]

    return " ".join(words).lower()

df["text"] = df["text"].apply(clean_text)

diseases = df["label"].unique()

# helper dictionaries to convert diseases to index and vice versa
idx2dis = {k:v for k,v in enumerate(diseases)}
dis2idx = {v:k for k,v in idx2dis.items()}

# convert disease name to index (label encoding)
df["label"] = df["label"].apply(lambda x: dis2idx[x])

# Split the data into train,test set
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# pytorch dataset object use index to return item, so need to reset non-continuoues index of divided dataset
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

max_words = X_train.apply(lambda x:x.split()).apply(len).max()

counter = Counter()
for text in X_train:
    counter.update(text.split())

vocab = torchtext.vocab.build_vocab_from_iterator(counter,specials=['<unk>', '<pad>'])
# set default index as unknown token
vocab.set_default_index(vocab['<unk>'])


# Create a PyTorch dataset`
class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, symptoms, labels):
        self.symptoms = symptoms
        self.labels = torch.tensor(labels.to_numpy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.symptoms[idx]
        label = self.labels[idx]

        # Convert the text to a sequence of word indices
        text_indices = [vocab[word] for word in text.split()]

        # padding for same length sequence
        if len(text_indices) < max_words:
            text_indices = text_indices + [1] * (max_words - len(text_indices))

        return torch.tensor(text_indices), label

# instantiate dataset objects
train_dataset = DiseaseDataset(X_train, y_train)
val_dataset = DiseaseDataset(X_test, y_test)

# choose batch size, will start from smaller values as we got smaller dataset
batch_size = 1200

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Define the RNN model
class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, drop_prob, num_layers=1, bidir=False,
                 seq="lstm"):
        super(RNNModel, self).__init__()
        self.seq = seq
        self.bidir_f = 2 if bidir else 0
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=bidir)

        self.dropout = torch.nn.Dropout(drop_prob)  # dropout layer
        self.fc = torch.nn.Linear(hidden_dim * self.bidir_f, num_classes)  # fully connected layer

    def forward(self, text_indices):
        # Embed the text indices
        embedded_text = self.embedding(text_indices)
        #         print("EMB SHAPE: ",embedded_text.shape)

        # Pass the embedded text through the RNN
        rnn_output, hidden_states = self.rnn(embedded_text)
        # Take the last output of the RNN
        last_rnn_output = rnn_output[:, -1, :]
        x = self.dropout(last_rnn_output)
        # Pass the last output of the RNN through the fully connected layer
        x = self.fc(x)

        # Return the final output
        return x


def train(model, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # choose device for training
    device = "cpu"
    model = model.to(device)
    print("IS CUDA: ", next(model.parameters()).is_cuda)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (labels == outputs.argmax(dim=-1)).float().mean().item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = outputs.argmax(-1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (labels == outputs.argmax(dim=-1)).float().mean().item()
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss}, Train Accuracy: {acc:.2f}  Val Accuracy: {accuracy:.2f}')

num_classes = len(np.unique(y_train))
vocab_size = len(vocab)
emb_dim = 256
hidden_dim = 128
drop_prob = 0.4



model_gru = RNNModel(vocab_size,emb_dim,hidden_dim,num_classes,drop_prob,num_layers=1,bidir=True,seq="gru")

train(model_gru,20)


def make_pred(model, text):
    text = clean_text(text)
    # Convert the text to a sequence of word indices
    text_indices = [vocab[word] for word in text.split()]

    # padding for same length sequence
    if len(text_indices) < max_words:
        text_indices = text_indices + [1] * (max_words - len(text_indices))
    text_indices = torch.tensor(text_indices).to('cpu')
    pred = model(text_indices.unsqueeze(0))

    return (idx2dis[pred.argmax(1).item()])

symp2 = "I've been itching a lot, and it's been accompanied with a rash that looks to be getting worse over time. \
There are also some patches of skin that are different colours from the rest of the skin,\
as well as some lumps that resemble little nodes."


torch.save(model_gru.state_dict(), r'C:\Users\dell\PycharmProjectfx\pythonProject\modelx\modex.pt')


num_classes = len(np.unique(y_train))
vocab_size = len(vocab)
emb_dim = 256
hidden_dim = 128
drop_prob = 0.4



model_gru.load_state_dict(torch.load(r'C:\Users\dell\PycharmProjectfx\pythonProject\modelx\modex.pt'))
model_gru.eval()
make_pred(model_gru, symp2)