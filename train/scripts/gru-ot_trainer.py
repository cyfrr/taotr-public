import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU available.")
else:
    print("GPU not available.")

# Create 'gru_weights' directory if it doesn't exist
if not os.path.exists('../gru_weights'):
    os.makedirs('../gru_weights')

# Load dataset
data_path = "../../datasets/msft special/msft_optimal_intraday_data.csv"
df = pd.read_csv(data_path)

# Preprocess data
scaler_close = MinMaxScaler(feature_range=(-1, 1))
scaler_optimal = MinMaxScaler(feature_range=(-1, 1))

# Scale close and optimal columns
scaled_close = scaler_close.fit_transform(df['close'].values.reshape(-1, 1))
scaled_optimal = scaler_optimal.fit_transform(df['optimal'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 100
X, y = create_sequences(scaled_close, scaled_optimal, seq_length)

# Dataset class
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = StockDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Training settings
input_size = 1
hidden_size = 90
output_size = 1
num_layers = 4
learning_rate = 0.001
num_epochs = 300

model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model():
    for epoch in range(num_epochs):
        for i, (seq, target) in enumerate(dataloader):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    save_path = '../gru_weights/gru_ot.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

train_model()

# Load the previously saved model
model_load_path = '../gru_weights/gru_ot.pth'
model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode
print(f'Model loaded from {model_load_path}')

# Function to make predictions
def predict(model, data, seq_length):
    model.eval()
    predictions = []
    for i in range(len(data) - seq_length):
        seq = torch.tensor(data[i:i + seq_length], dtype=torch.float32).to(device)
        seq = seq.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            prediction = model(seq)
            predictions.append(prediction.item())
    return predictions

# Get predictions
predicted_optimal = predict(model, scaled_close, seq_length)

# Adjust the length of actual optimal values to align with predicted values
actual_optimal = df['optimal']
aligned_actual_optimal = actual_optimal[seq_length:]
aligned_actual_optimal = aligned_actual_optimal[:len(predicted_optimal)]

# Inverse transform the predictions
predicted_optimal_values = scaler_optimal.inverse_transform(np.array(predicted_optimal).reshape(-1, 1))

# Plotting actual vs. predicted optimal values
plt.figure(figsize=(10, 5))
plt.plot(aligned_actual_optimal, label='Actual Optimal')
plt.plot(predicted_optimal_values, label='Predicted Optimal')
plt.legend()
plt.title('Actual vs Predicted Optimal Values')
plt.xlabel('Time')
plt.ylabel('Optimal Value')
plt.savefig('gru_optimal_prediction_vs_actual.png')
plt.show()