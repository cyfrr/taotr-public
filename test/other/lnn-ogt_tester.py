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
from ncps.torch import LTC
from ncps.wirings import AutoNCP

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


input_size = 1
units = 360
output_size = 1
LNNModel = LTC(input_size, AutoNCP(units,output_size))


class EnhancedLNNBacktester:
    def __init__(self, df, model_path, initial_capital=100000, max_drawdown_pct=25):
        self.df = df

        # Model parameters
        input_size = 1
        hidden_size = 90
        output_size = 1
        num_layers = 4

        # Initialize and load model
        self.model = LNNModel.to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # Trading parameters
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.lookback = 100
        self.position_size = 0.95
        self.prediction_threshold = 0.01  # 0.1% threshold

        # Risk management
        self.active = True
        self.peak_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.drawdown_halt = False
        self.halt_reason = None

        # Create a MinMaxScaler and fit it to the scaler dataset
        scaler_path = "../../datasets/msft_1h_intraday_data.csv"
        scaler_df = pd.read_csv(scaler_path)
        scaler_df['date'] = pd.to_datetime(scaler_df['date'])
        scaler_df.set_index("date", inplace=True)
        scaler_df.columns = [col.capitalize() for col in scaler_df.columns]

        self.scaler = MinMaxScaler()
        self.scaler.fit(scaler_df[['Close']])

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
            hist_prices = self.df['Close'].iloc[i - self.lookback:i].values
            current_price = self.df['Close'].iloc[i]

            # Get model's prediction and calculate signal
            x = self.prepare_data(hist_prices)
            with torch.no_grad():
                prediction = self.model(x)
                predicted_next_price = prediction.item()

                # Ensure the predicted_next_price matches the scaler's input shape
                predicted_next_price_scaled = self.scaler.inverse_transform([[predicted_next_price]])[0][0]

            # Calculate predicted return and normalize to [-1, 1] signal
            predicted_return = (predicted_next_price_scaled - current_price) / current_price
            signal = np.clip(predicted_return / self.prediction_threshold, -1, 1)

            # Check drawdown and capital
            if self.check_drawdown() and self.active:
                print(f"\nTrading halted at {self.df.index[i]} due to excessive drawdown")

            if self.capital <= 0 and self.active:
                print(f"\nTrading halted at {self.df.index[i]} due to negative capital")
                self.active = False
                self.halt_reason = "Negative capital"

            # Close positions if we're inactive
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

            # Trading logic
            if self.active:
                target_position_value = self.capital * self.position_size * signal
                target_position = target_position_value / current_price

                # If we need to adjust position
                if abs(target_position - self.position) > 0.01:
                    # Close existing position if we have one
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

                    # Open new position if signal is strong enough
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

            # Calculate current drawdown for results
            current_drawdown_pct = ((
                                            self.peak_capital - self.capital) / self.peak_capital) * 100 if self.peak_capital > 0 else 0

            results.append({
                'timestamp': self.df.index[i],
                'price': current_price,
                'predicted_next_price': predicted_next_price_scaled,
                'predicted_return': predicted_return,
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
        fig = make_subplots(rows=4, cols=1,
                            subplot_titles=('Trading Strategy Capital Over Time',
                                            'Drawdown %',
                                            'Model Signal and Predicted Returns',
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

        fig.add_hline(y=-self.max_drawdown_pct,
                      line_dash="dash",
                      line_color="red",
                      annotation_text=f"Max Drawdown ({self.max_drawdown_pct}%)",
                      row=2, col=1)

        # Plot signals and predicted returns
        fig.add_trace(
            go.Scatter(x=results_df['timestamp'],
                       y=results_df['signal'],
                       name='Model Signal',
                       line=dict(color='purple')),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=results_df['timestamp'],
                       y=results_df['predicted_return'],
                       name='Predicted Return',
                       line=dict(color='orange', dash='dot')),
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

        # Add trade markers
        for action, color, symbol in [
            ('long', 'green', 'triangle-up'),
            ('short', 'red', 'triangle-down'),
            ('close', 'grey', 'x'),
            ('forced_close', 'black', 'x')
        ]:
            mask = trades_df['action'] == action
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


# Example usage:
if __name__ == "__main__":
    # Load dataset
    data_path = "../../datasets/testing datasets/aapl_1h_firstratedata.csv"
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    df.columns = [col.capitalize() for col in df.columns]

    scaler_path = "../../datasets/msft_1h_intraday_data.csv"

    # Load scaler dataset
    scaler_df = pd.read_csv(scaler_path)
    scaler_df['date'] = pd.to_datetime(scaler_df['date'])
    scaler_df.set_index("date", inplace=True)
    scaler_df.columns = [col.capitalize() for col in scaler_df.columns]

    # Create a MinMaxScaler and fit it to the scaler dataset
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(scaler_df)


    # Function to reverse scale the values
    def reverse_scale(scaled_values):
        return scaler.inverse_transform(scaled_values)

    # Initialize and run backtest
    model_path = "../../train/gru_weights/gru_independent.pth"
    backtest = EnhancedLNNBacktester(df, model_path, initial_capital=100000, max_drawdown_pct=25)
    results_df, trades_df = backtest.run_backtest()
    backtest.create_visualizations(results_df, trades_df)