import pandas as pd
import torch
import numpy as np
from datetime import timedelta
import os
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from ncps.wirings import AutoNCP
from ncps.torch import CfC

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load dataset
data_path = "../../datasets/testing datasets/msft_1h_firstratedata.csv"
df = pd.read_csv(data_path)

df['date'] = pd.to_datetime(df['date'])
df.set_index("date", inplace=True)

df.columns = [col.capitalize() for col in df.columns]

class LNNBacktester:
    def __init__(self, df, model_path, initial_capital=100000):
        self.df = df

        # Initialize scaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(df['Close'].values.reshape(-1, 1))

        # Define the model parameters
        input_size = 1
        self.wiring = AutoNCP(units=25, output_size=1)

        # Initialize the model
        self.model = CfC(input_size=input_size, units=self.wiring)

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
        self.lookback = 100  # Same as seq_length in training
        self.prediction_threshold = 0.001
        self.position_size = 1

    def prepare_data(self, prices):
        # Scale the data
        scaled_prices = self.scaler.transform(prices.reshape(-1, 1))

        # Convert to tensor and reshape for LNN
        x = torch.FloatTensor(scaled_prices).to(device)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for LNN input
        return x

    def run_backtest(self):
        results = []
        total_steps = len(self.df) - self.lookback

        pbar = tqdm(range(self.lookback, len(self.df)),
                    desc="Running backtest",
                    total=total_steps)

        for i in pbar:
            # Get historical prices for the lookback period
            hist_prices = self.df['Close'].iloc[i - self.lookback:i].values
            current_price = self.df['Close'].iloc[i]

            # Prepare data and get prediction
            x = self.prepare_data(hist_prices)
            with torch.no_grad():
                prediction = self.model(x)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]  # Extract actual output tensor

                # Convert prediction back to price scale
                predicted_next_price = self.scaler.inverse_transform(
                    prediction.cpu().numpy().reshape(-1, 1)
                )[0][0]

            # Calculate predicted return as percentage
            predicted_return = (predicted_next_price - current_price) / current_price

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
            title_text="LNN Trading Strategy Analysis"
        )

        fig.show()
model_path = "path/to/lnn_weights/lnn_independent.pth"
backtest = LNNBacktester(df, model_path)
results_df, trades_df = backtest.run_backtest()

# Print summary
print(f"\nFinal Capital: ${backtest.capital:,.2f}")
print(f"Total Trades: {len(trades_df)}")
print(f"Total Return: {((backtest.capital - 100000) / 100000 * 100):.2f}%")

# Create visualizations
backtest.create_visualizations(results_df, trades_df)