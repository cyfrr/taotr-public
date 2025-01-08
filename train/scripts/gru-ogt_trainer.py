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
data_path = "../../datasets/msft special/msft_mentor_intraday_data.csv"
df = pd.read_csv(data_path)

optimal_data_path = "../../datasets/msft special/msft_optimal_intraday_data.csv"
optimal_df = pd.read_csv(optimal_data_path)

# Preprocess data
scaler_close = MinMaxScaler(feature_range=(-1, 1))
scaler_mentor = MinMaxScaler(feature_range=(-1, 1))
scaler_optimal = MinMaxScaler(feature_range=(-1, 1))

# Scale close, mentor, and optimal columns
scaled_close = scaler_close.fit_transform(df['close'].values.reshape(-1, 1))
scaled_mentor = scaler_mentor.fit_transform(df['mentor'].values.reshape(-1, 1))
scaled_optimal = scaler_optimal.fit_transform(optimal_df['optimal'].values.reshape(-1, 1))


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
X, y_mentor = create_sequences(scaled_close, scaled_mentor, seq_length)
_, y_optimal = create_sequences(scaled_close, scaled_optimal, seq_length)


# Dataset class with dynamic target
class StockDataset(Dataset):
    def __init__(self, X, y, transition_weight=0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_mentor = torch.tensor(y[0], dtype=torch.float32)
        self.y_optimal = torch.tensor(y[1], dtype=torch.float32)
        self.transition_weight = transition_weight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Interpolate between mentor and optimal targets
        y = (1 - self.transition_weight) * self.y_mentor[idx] + self.transition_weight * self.y_optimal[idx]
        return self.X[idx], y


# GRU Model (same as before)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Training settings
input_size = 1
hidden_size = 100
output_size = 1
num_layers = 320
learning_rate = 0.001
num_epochs = 720

model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

def calculate_transition_weight(epoch, total_epochs):
    # Use a smooth sigmoid-like transition
    transition_start = total_epochs * 0.33  # Start transition at 1/3 of total epochs
    transition_end = total_epochs * 0.66  # End transition at 2/3 of total epochs

    # Smooth transition using a logistic function
    if epoch <= transition_start:
        return 0
    elif epoch >= transition_end:
        return 1
    else:
        # Logistic function for smooth transition
        x = (epoch - transition_start) / (transition_end - transition_start) * 10 - 5
        return 1 / (1 + np.exp(-x))

# Plot transition weights
total_epochs = num_epochs
transition_weights = [calculate_transition_weight(epoch, total_epochs) for epoch in range(total_epochs)]

plt.figure(figsize=(10, 5))
plt.plot(range(total_epochs), transition_weights)
plt.title('Transition Weight over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Transition Weight')
plt.show()

# Modify training loop
def train_model():
    for epoch in range(num_epochs):
        # Calculate smooth transition weight
        transition_weight = calculate_transition_weight(epoch, num_epochs)

        # Create dataset with current transition weight
        dataset = StockDataset(X, [y_mentor, y_optimal], transition_weight)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Rest of the training loop remains the same
        for i, (seq, target) in enumerate(dataloader):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Transition Weight: {transition_weight:.4f}')

    # Save the model
    save_path = '../gru_weights/gru_ogt_xl.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


# Train the model
train_model()


# Prediction function (similar to before)
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


# Get predictions using the last saved model
predicted_optimal = predict(model, scaled_close, seq_length)

# Adjust the length of actual optimal values to align with predicted values
actual_optimal = optimal_df['optimal']
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
plt.savefig('gru_ogt_prediction_xl_vs_actual.png')
plt.show()