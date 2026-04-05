import torch
import torch.nn as nn
import random
import numpy as np

import keras
from tensorflow.keras.datasets import imdb
from keras.preprocessing import sequence
from keras.datasets import imdb
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#Define Max_Features
MAX_FEATURES = 1000

#Load IMDB datasets
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=MAX_FEATURES)

#Pad sequences with maxlen 100
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

#Create LSTM
class LSTMClassifier(nn.Module):
     def __init__(self, MAX_FEATURES, embedding_dim, hidden_dim, num_layers):
         super(LSTMClassifier, self).__init__()
         self.embeddings = nn.Embedding(MAX_FEATURES,embedding_dim)
         self.hidden_dim = hidden_dim
         self.num_layers = num_layers
         self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers,batch_first=True)
         self.classifier = nn.Linear(hidden_dim,1)

     def forward(self, x):
        # 1. Convert word indices to vectors
        # x shape: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        x = self.embeddings(x)

        batch_sz, seq_sz, _ = x.size()
        # 2. Initialize hidden state with zeros
        h_t = torch.zeros(self.num_layers, batch_sz, self.hidden_dim).to(x.device)
        c_t = torch.zeros(self.num_layers, batch_sz, self.hidden_dim).to(x.device)

        #3. Forward Pass
        output,(h_t,c_t) = self.lstm(x, (h_t,c_t))
        h_t = h_t[-1]

        
        # 4. Final Classification
        # Use the very last hidden state (h_t) because it contains the summary of the whole sentence.
        prediction = self.classifier(h_t)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze(1)

        return prediction


#initialize LSTM tools for training
LSTM_model = LSTMClassifier(MAX_FEATURES, 8,8,8)
BCE = nn.BCELoss()
optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=0.01)

#Turn training arrays to tensors
x_train = torch.tensor(x_train).long()
y_train = torch.tensor(y_train).float()

#Turn Testing arrays to Tensors
x_test = torch.tensor(x_test).long()
y_test = torch.tensor(y_test).float()

#Wrap Tensors into Dataset
Dataset = TensorDataset(x_train, y_train)
ValDataset = TensorDataset(x_test, y_test)

#Create Dataloaders

train_loader = DataLoader(
    dataset=Dataset,      # Dataset object
    batch_size=64,              # Samples per iteration
    shuffle=True,               # Reshuffle at every epoch
)

validation_loader = DataLoader(
    dataset=ValDataset,      # Dataset object
    batch_size=64,              # Samples per iteration
    shuffle=True,               # Reshuffle at every epoch
)

Accuracy = 0
Best_Accuracy = 0

#Training Loop
for epoch in range(10):
  Accuracy = 0
  validation_loss = 0
  for inputs, targets in train_loader:
    Predictions = LSTM_model(inputs)
    loss = BCE(Predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

  

  #Validation Loop
  for(inputs, targets) in validation_loader:
    Predictions = LSTM_model(inputs)
    loss = BCE(Predictions, targets)
    validation_loss+=loss.item()
    Predictions = torch.round(Predictions)
    
     #Count Correct Predictions
    Accuracy += (Predictions == targets).sum().item()

    #Calculate Accuracy
  Accuracy = Accuracy/len(x_test)
  if(Accuracy > Best_Accuracy):
    Best_Accuracy = Accuracy

  print(f"Epoch {epoch}, Validation Loss:{validation_loss}")
  
    

  

print(f" FINAL LOSS: Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
print(f"Best Accuracy: {Best_Accuracy}")

