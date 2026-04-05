from sklearn.externals._packaging.version import PrePostDevType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error

#Load the CSV file data
try:
  df = pd.read_csv('test.txt')
except FileNotFoundError:
  print("Error: File 'test.txt' not found.")
  exit()

try:
  df2 = pd.read_csv('train.txt')
except FileNotFoundError:
  print("Error: File 'train.txt' not found.")
  exit()

X_train_df = df2.copy()
X_test_df = df.copy()

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize training and testing data
# Only fit the training data so the model learns training min/max
X_train_scaled = scaler.fit_transform(X_train_df[['Passengers']])
X_test_scaled = scaler.transform(X_test_df[['Passengers']])

# Print out the shape of the data
print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of X_test_scaled:", X_test_scaled.shape)

# Function to create sequences for RNN inputs
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Define sequence length (look-back window)
sequence_length = 10

# Create sequences for training data
X_train_sequences, y_train_labels = create_sequences(X_train_scaled, sequence_length)
X_test_sequences, y_test_labels = create_sequences(X_test_scaled, sequence_length)

class RNN(nn.Module):
   def __init__(self, input_size, hidden_size, output_size):
       super(RNN, self).__init__()
       self.hidden_size = hidden_size
       self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
       self.fc = nn.Linear(hidden_size, output_size)


   def forward(self, x):
    # x goes through the RNN
        out, _ = self.rnn(x)

        # Only take the last time step's output to predict the future
        out = out[:, -1, :]

        # Pass that last step through the Linear layer
        prediction = self.fc(out)
        return prediction

#Initialize model, loss fucntion, and optimizer
# input_size should be the number of features at each time step, which is 1 for 'Passengers'
RNN_model = RNN(input_size=1, hidden_size=4, output_size=1)
MSE = nn.MSELoss()
optimizer = torch.optim.Adam(RNN_model.parameters(), lr=0.01)

#Training Loop
for epoch in range(1000):

  #Set up the data
  inputs = torch.from_numpy(X_train_sequences).float()
  targets = torch.from_numpy(y_train_labels).float()

  #Make Predictions
  Predictions = RNN_model(inputs)
  loss = MSE(Predictions, targets)

  #Adjust Optimizer
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if(epoch % 100 == 0):
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}\n")

#Testing Loop
for epoch in range(1000):
  test_inputs = torch.from_numpy(X_test_sequences).float()
  test_targets = torch.from_numpy(y_test_labels).float()

  Test_Predictions = RNN_model(test_inputs)
  Loss = MSE(Test_Predictions, test_targets)

  optimizer.zero_grad()
  Loss.backward()
  optimizer.step()

  if(epoch == 0):
    print("TESTING LOOP")

  if(epoch % 100 == 0):
    print(f"Epoch {epoch}, Loss: {Loss.item():.4f}")

print(f"TESTING FINAL RESULT: Epoch {epoch+1}, Loss: {Loss.item():.4f}\n")

#Denormalize the data for viewing
denormalized_predictions = scaler.inverse_transform(Test_Predictions.detach().numpy())
denormalized_targets = scaler.inverse_transform(test_targets.detach().numpy())

#Calculate and show RMSE
RMSE = root_mean_squared_error(denormalized_targets, denormalized_predictions)
print(f"Root Mean Squared Error: {RMSE}")

#Plot Testing Predictions agaisnt Testing targets
plt.figure(figsize=(12, 6))
plt.plot(denormalized_targets, label='Actual', color='blue')
plt.plot(denormalized_predictions, label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.title('Actual vs. Predicted Passengers')
plt.legend()
plt.show
