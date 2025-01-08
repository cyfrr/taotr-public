import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datetime import timedelta
import os
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Load dataset
data_path = "../../datasets/testing datasets/MSFT_5min_firstratedata.csv"
df = pd.read_csv(data_path)

df['date'] = pd.to_datetime(df['date'])
df.set_index("date", inplace=True)

df.columns = [col.capitalize() for col in df.columns]

scalar_data_path = "../../datasets/msft_5m_intraday_data.csv"
scalar_df = pd.read_csv(data_path)
scaler = MinMaxScaler()
scaler.fit(scalar_df[['close']].values)

class GRUBacktester:
    def __init__(self, df, model_path, initial_capital=100000):
        # Store the DataFrame as instance variable
        self.df = df

        # Define the model parameters
        input_size = 1
        hidden_size = 64
        output_size = 1
        num_layers = 2

        # Initialize the model
        self.model = GRUModel(input_size, hidden_size, output_size, num_layers)

        # Load the model weights
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.capital = initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []

        # Parameters
        self.lookback = 100
        self.prediction_threshold = 0.01  # 0.1% threshold
        self.position_size = 0.95

    def prepare_data(self, prices):
        x = torch.FloatTensor(prices).unsqueeze(1)
        x = (x - x.mean()) / x.std()
        x = x.unsqueeze(0)
        return x.to(device)

    def run_backtest(self):
        results = []
        total_steps = len(self.df) - self.lookback

        pbar = tqdm(range(self.lookback, len(self.df)),
                    desc="Running backtest",
                    total=total_steps)

        for i in pbar:
            # Get historical prices for the lookback period
            hist_prices = self.df['Close'].iloc[i - self.lookback:i].values

            # Prepare data and get prediction
            x = self.prepare_data(hist_prices)
            with torch.no_grad():
                prediction = self.model(x)
                predicted_next_price = prediction.item()

                # Reverse MinMaxScaler for the predicted_next_price
                predicted_next_price = scaler.inverse_transform(np.array(prediction.cpu()).reshape(-1, 1)).item()
            current_price = self.df['Close'].iloc[i]

            # Calculate predicted return as percentage
            predicted_return = (predicted_next_price - current_price)/predicted_next_price

            # Trading logic based on predicted return
            if predicted_return > self.prediction_threshold:  # Predicted upward move
                if self.position <= 0:  # If no position or short position
                    if self.position < 0:  # Close short if exists
                        profit = self.position_value - (abs(self.position) * current_price)
                        self.capital += profit
                        self.trades.append({
                            'timestamp': self.df.index[i],
                            'action': 'close_short',
                            'price': current_price,
                            'profit': profit
                        })

                    # Open long position
                    self.position = (self.capital * self.position_size) / current_price
                    self.position_value = self.position * current_price
                    self.trades.append({
                        'timestamp': self.df.index[i],
                        'action': 'long',
                        'price': current_price,
                        'size': self.position
                    })

            elif predicted_return < -self.prediction_threshold:  # Predicted downward move
                if self.position >= 0:  # If no position or long position
                    if self.position > 0:  # Close long if exists
                        profit = (self.position * current_price) - self.position_value
                        self.capital += profit
                        self.trades.append({
                            'timestamp': self.df.index[i],
                            'action': 'close_long',
                            'price': current_price,
                            'profit': profit
                        })

                    # Open short position
                    self.position = -(self.capital * self.position_size) / current_price
                    self.position_value = abs(self.position) * current_price
                    self.trades.append({
                        'timestamp': self.df.index[i],
                        'action': 'short',
                        'price': current_price,
                        'size': self.position
                    })

            else:  # Predicted move within threshold (neutral)
                if self.position != 0:  # Close any existing position
                    if self.position > 0:
                        profit = (self.position * current_price) - self.position_value
                    else:
                        profit = self.position_value - (abs(self.position) * current_price)

                    self.capital += profit
                    self.trades.append({
                        'timestamp': self.df.index[i],
                        'action': 'close',
                        'price': current_price,
                        'profit': profit
                    })
                    self.position = 0
                    self.position_value = 0

            results.append({
                'timestamp': self.df.index[i],
                'price': current_price,
                'predicted_next_price': predicted_next_price,
                'predicted_return': predicted_return,
                'position': self.position,
                'capital': self.capital
            })

            pbar.set_postfix({'Capital': f'${self.capital:,.2f}'})

        return pd.DataFrame(results), pd.DataFrame(self.trades)

    def create_visualizations(self, results_df, trades_df):
        # Create subplots
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Trading Strategy Capital Over Time',
                                            'Stock Price with Trade Indicators'),
                            vertical_spacing=0.15)

        # Plot capital over time
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'],
                       y=results_df['capital'],
                       name='Capital',
                       line=dict(color='blue')),
            row=1, col=1
        )

        # Plot stock price
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'],
                       y=results_df['price'],
                       name='Stock Price',
                       line=dict(color='black')),
            row=2, col=1
        )

        # Add trade markers
        long_entries = trades_df[trades_df['action'] == 'long']
        short_entries = trades_df[trades_df['action'] == 'short']
        closes = trades_df[trades_df['action'].str.startswith('close')]

        # Plot long entries
        fig.add_trace(
            go.Scatter(x=long_entries['timestamp'],
                       y=long_entries['price'],
                       mode='markers',
                       name='Long Entry',
                       marker=dict(color='green', symbol='triangle-up', size=8)),
            row=2, col=1
        )

        # Plot short entries
        fig.add_trace(
            go.Scatter(x=short_entries['timestamp'],
                       y=short_entries['price'],
                       mode='markers',
                       name='Short Entry',
                       marker=dict(color='red', symbol='triangle-down', size=8)),
            row=2, col=1
        )

        # Plot closes
        fig.add_trace(
            go.Scatter(x=closes['timestamp'],
                       y=closes['price'],
                       mode='markers',
                       name='Position Close',
                       marker=dict(color='grey', symbol='x', size=8)),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Trading Strategy Analysis"
        )

        fig.show()


# Usage example:
model_path = "../../train/gru_weights/gru_independent.pth"
backtest = GRUBacktester(df, model_path)
results_df, trades_df = backtest.run_backtest()

# Print summary
print(f"\nFinal Capital: ${backtest.capital:,.2f}")
print(f"Total Trades: {len(trades_df)}")
print(f"Total Return: {((backtest.capital - 100000) / 100000 * 100):.2f}%")

# Create visualizations
backtest.create_visualizations(results_df, trades_df)