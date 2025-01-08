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
data_path = "../../datasets/msft_1h_intraday_data.csv"
df = pd.read_csv(data_path)

# Preprocess data
scaler = MinMaxScaler(feature_range=(-1, 1))
print(df.columns)
scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))


# Create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 100
X, y = create_sequences(scaled_data, seq_length)


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
    save_path = '../gru_weights/gru_independent.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

train_model()

# Load the previously saved model
model_load_path = '../gru_weights/gru_independent.pth'
model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode
print(f'Model loaded from {model_load_path}')

# Testing the GRU model
model.eval()  # Set the model to evaluation mode


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
predicted_prices = predict(model, scaled_data, seq_length)
actual_prices = df['close']

# Adjust the length of actual prices to align with predicted prices
aligned_actual_prices = actual_prices[seq_length:]
aligned_actual_prices = aligned_actual_prices[:len(predicted_prices)]

# Scale the predicted prices to the same range as the actual prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Plotting actual vs. predicted prices
plt.figure(figsize=(10, 5))
plt.plot(aligned_actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.legend()
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.savefig('gru_prediction_vs_actual.png')
plt.show()