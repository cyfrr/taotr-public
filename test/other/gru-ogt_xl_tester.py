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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


# Load dataset
data_path = "../../datasets/testing datasets/meta_1h_firstratedata.csv"
df = pd.read_csv(data_path)

df['date'] = pd.to_datetime(df['date'])
df.set_index("date", inplace=True)

df.columns = [col.capitalize() for col in df.columns]



class PositionSignalBacktester:
        def __init__(self, df, model_path, train_data_path=None, initial_capital=100000, max_drawdown_pct=25):
            self.df = df

            self.input_size = 1  # Close, Position, and Previous Trades
            self.hidden_size = 100
            self.output_size = 1
            self.num_layers = 320

            self.model = GRUModel(self.input_size, self.hidden_size, self.output_size, self.num_layers)
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()

            # Initialize price scaler
            self.price_scaler = MinMaxScaler()
            self.price_scaler.fit(df[['Close']].values)

            # Initialize signal scaler
            self.signal_scaler = MinMaxScaler()
            if train_data_path:
                train_df = pd.read_csv(train_data_path)
                # Assuming the training data has a 'target' column that was used for training
                # Adjust this based on your actual training data structure
                self.signal_scaler.fit(train_df[['target']].values)
            else:
                # If no training data provided, use a default range of [-1, 1]
                self.signal_scaler.fit(np.array([-1, 1]).reshape(-1, 1))

            # Trading parameters
            self.initial_capital = initial_capital
            self.capital = initial_capital
            self.position = 0
            self.position_value = 0
            self.trades = []
            self.lookback = 100
            self.position_size = 0.95
            self.active = True

            # Drawdown tracking
            self.peak_capital = initial_capital
            self.max_drawdown_pct = max_drawdown_pct
            self.drawdown_halt = False
            self.halt_reason = None

        def check_drawdown(self):
            """Check if current drawdown exceeds maximum allowed"""
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital

            current_drawdown_pct = ((self.peak_capital - self.capital) / self.peak_capital) * 100

            if current_drawdown_pct > self.max_drawdown_pct and not self.drawdown_halt:
                self.drawdown_halt = True
                self.active = False
                self.halt_reason = f"Maximum drawdown of {self.max_drawdown_pct}% exceeded"
                return True
            return False

        def prepare_data(self, prices, positions, trades):
            # Scale the inputs
            scaled_prices = self.price_scaler.transform(prices.reshape(-1, 1))

            # Combine features
            x = np.array(scaled_prices)
            x = torch.FloatTensor(x).unsqueeze(0)
            return x.to(device)

        def scale_signal(self, value):

            # Original range
            x_min = -0.4
            x_max = 0.2

            # Target range
            y_min = 0
            y_max = 0.8

            return ((value - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min

        def denormalize_signal(self, signal):
            signal = np.clip(signal, -1, 1)  # Clip to valid range
            signal_2d = np.array([[signal]])
            denorm_signal = self.signal_scaler.inverse_transform(signal_2d)[0, 0]
            return denorm_signal

        def run_backtest(self):
            results = []
            positions_hist = np.zeros(self.lookback)
            trades_hist = np.zeros(self.lookback)

            total_steps = len(self.df) - self.lookback
            pbar = tqdm(range(self.lookback, len(self.df)), desc="Running backtest", total=total_steps)

            for i in pbar:
                hist_prices = self.df['Close'].iloc[i - self.lookback:i].values
                current_price = self.df['Close'].iloc[i]

                x = self.prepare_data(hist_prices, positions_hist, trades_hist)
                with torch.no_grad():
                    signal = self.model(x).item()
                    # Denormalize the signal
                    signal = self.scale_signal(signal)

                positions_hist = np.roll(positions_hist, -1)
                positions_hist[-1] = self.position
                trades_hist = np.roll(trades_hist, -1)
                trades_hist[-1] = 0

                if self.check_drawdown() and self.active:
                    print(f"\nTrading halted at {self.df.index[i]} due to excessive drawdown")

                if self.capital <= 0 and self.active:
                    print(f"\nTrading halted at {self.df.index[i]} due to negative capital")
                    self.active = False
                    self.halt_reason = "Negative capital"

                if not self.active and self.position != 0:
                    if self.position > 0:
                        profit = (self.position * current_price) - self.position_value
                    else:
                        profit = self.position_value - (abs(self.position) * current_price)

                    self.capital += profit
                    self.trades.append({
                        'timestamp': self.df.index[i],
                        'action': 'forced_close',
                        'price': current_price,
                        'profit': profit,
                        'signal': signal,
                        'capital': self.capital,
                        'reason': self.halt_reason
                    })
                    self.position = 0
                    self.position_value = 0
                    trades_hist[-1] = 1

                if self.active:
                    target_position_value = self.capital * self.position_size * signal
                    target_position = target_position_value / current_price

                    if abs(target_position - self.position) > 0.000001:
                        if self.position != 0:
                            if self.position > 0:
                                profit = (self.position * current_price) - self.position_value
                            else:
                                profit = self.position_value - (abs(self.position) * current_price)

                            self.capital += profit
                            self.trades.append({
                                'timestamp': self.df.index[i],
                                'action': 'close',
                                'price': current_price,
                                'profit': profit,
                                'signal': signal,
                                'capital': self.capital
                            })
                            trades_hist[-1] = 1

                        if abs(signal) > 0.1 and self.capital > 1000:
                            self.position = target_position
                            self.position_value = abs(self.position) * current_price

                            self.trades.append({
                                'timestamp': self.df.index[i],
                                'action': 'long' if signal > 0 else 'short',
                                'price': current_price,
                                'size': self.position,
                                'signal': signal,
                                'capital': self.capital
                            })
                            trades_hist[-1] = 1

                current_drawdown_pct = ((self.peak_capital - self.capital) / self.peak_capital) * 100 if self.peak_capital > 0 else 0

                results.append({
                    'timestamp': self.df.index[i],
                    'price': current_price,
                    'signal': signal,
                    'position': self.position,
                    'capital': self.capital,
                    'peak_capital': self.peak_capital,
                    'drawdown_pct': current_drawdown_pct,
                    'active': self.active
                })

                pbar.set_postfix({
                    'Capital': f'${self.capital:,.2f}',
                    'Active': self.active,
                    'Drawdown': f'{current_drawdown_pct:.1f}%'
                })

            return pd.DataFrame(results), pd.DataFrame(self.trades)

        def create_visualizations(self, results_df, trades_df):
            # Check if trades_df is empty
            if trades_df.empty:
                print("No trades were executed during the backtest period.")
                trades_df = pd.DataFrame(columns=['timestamp', 'action', 'price', 'profit', 'signal', 'capital'])

            # Create subplots
            fig = make_subplots(rows=4, cols=1,
                                subplot_titles=('Trading Strategy Capital Over Time',
                                                'Drawdown %',
                                                'Model Signal Over Time',
                                                'Stock Price with Trade Indicators'),
                                vertical_spacing=0.1)

            # Plot capital over time
            fig.add_trace(
                go.Scatter(x=results_df['timestamp'],
                           y=results_df['capital'],
                           name='Capital',
                           line=dict(color='blue')),
                row=1, col=1
            )

            # Plot peak capital
            fig.add_trace(
                go.Scatter(x=results_df['timestamp'],
                           y=results_df['peak_capital'],
                           name='Peak Capital',
                           line=dict(color='green', dash='dash')),
                row=1, col=1
            )

            # Plot drawdown
            fig.add_trace(
                go.Scatter(x=results_df['timestamp'],
                           y=-results_df['drawdown_pct'],
                           name='Drawdown %',
                           line=dict(color='red')),
                row=2, col=1
            )

            # Add maximum drawdown line
            fig.add_hline(y=-self.max_drawdown_pct,
                          line_dash="dash",
                          line_color="red",
                          annotation_text=f"Max Drawdown ({self.max_drawdown_pct}%)",
                          row=2, col=1)

            # Plot signal over time
            fig.add_trace(
                go.Scatter(x=results_df['timestamp'],
                           y=results_df['signal'],
                           name='Model Signal',
                           line=dict(color='purple')),
                row=3, col=1
            )

            # Add signal threshold lines
            fig.add_hline(y=0.1, line_dash="dash", line_color="gray", row=3, col=1)
            fig.add_hline(y=-0.1, line_dash="dash", line_color="gray", row=3, col=1)

            # Plot stock price
            fig.add_trace(
                go.Scatter(x=results_df['timestamp'],
                           y=results_df['price'],
                           name='Stock Price',
                           line=dict(color='black')),
                row=4, col=1
            )

            # Add trade markers only if there are trades
            if not trades_df.empty:
                for action, color, symbol in [
                    ('long', 'green', 'triangle-up'),
                    ('short', 'red', 'triangle-down'),
                    ('close', 'grey', 'x'),
                    ('forced_close', 'black', 'x')
                ]:
                    mask = trades_df['action'].astype(str) == action
                    if mask.any():
                        fig.add_trace(
                            go.Scatter(x=trades_df[mask]['timestamp'],
                                       y=trades_df[mask]['price'],
                                       mode='markers',
                                       name=action.replace('_', ' ').title(),
                                       marker=dict(color=color, symbol=symbol, size=8)),
                            row=4, col=1
                        )

            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="Trading Strategy Analysis"
            )

            fig.show()

            # Print trading statistics
            print("\nTrading Statistics:")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Final Capital: ${self.capital:,.2f}")
            print(f"Peak Capital: ${self.peak_capital:,.2f}")
            print(f"Maximum Drawdown: {results_df['drawdown_pct'].max():.2f}%")
            print(f"Total Return: {((self.capital - self.initial_capital) / self.initial_capital * 100):.2f}%")
            print(f"Total Trades: {len(trades_df)}")
            print(f"Trading Status: {'Active' if self.active else 'Halted'}")
            if self.halt_reason:
                print(f"Halt Reason: {self.halt_reason}")

            if len(trades_df) > 0:
                profitable_trades = trades_df[trades_df['profit'] > 0]
                win_rate = len(profitable_trades) / len(trades_df) * 100
                print(f"Win Rate: {win_rate:.2f}%")

                avg_profit = trades_df[trades_df['profit'] > 0]['profit'].mean()
                avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean())
                if avg_loss > 0:
                    profit_factor = avg_profit / avg_loss
                    print(f"Profit Factor: {profit_factor:.2f}")


# Usage example:
model_path = "../../train/gru_weights/gru_ogt_xl.pth"
backtest = PositionSignalBacktester(df, model_path)
results_df, trades_df = backtest.run_backtest()

# Create visualizations
backtest.create_visualizations(results_df, trades_df)