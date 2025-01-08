import pandas as pd
import numpy as np


def generate_retrospective_optimal_signals(df, future_lookforward=25, profit_threshold=0.02):
    """
    Generate optimal trade signals by looking into the future to determine best possible trades

    Parameters:
    - df: Input DataFrame with price data
    - future_lookforward: Number of periods to look forward for trade evaluation
    - profit_threshold: Minimum profit percentage to trigger a strong signal

    Returns:
    - DataFrame with retrospective optimal trade signals
    """
    # Ensure the DataFrame is sorted chronologically
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Initialize optimal signal column
    df['optimal'] = 0.0

    # Calculate potential future price movements
    for i in range(len(df)):
        # Look forward to determine best possible trade
        if i + future_lookforward >= len(df):
            break

        # Calculate future prices and percentage changes
        future_prices = df['close'].iloc[i:i + future_lookforward + 1]
        max_price = future_prices.max()
        min_price = future_prices.min()
        current_price = df['close'].iloc[i]

        # Calculate percentage changes
        max_gain = (max_price - current_price) / current_price
        max_loss = (min_price - current_price) / current_price

        # Assign optimal signal based on potential future movements
        if max_gain > profit_threshold:
            # Strong buy signal if significant future gain possible
            df.loc[i, 'optimal'] = min(max_gain, 1.0)
        elif max_loss < -profit_threshold:
            # Strong sell signal if significant future loss possible
            df.loc[i, 'optimal'] = max(max_loss, -1.0)
        else:
            # Neutral signal if no significant movement expected
            df.loc[i, 'optimal'] = 0.0

    return df


def main():
    # Read the input data
    data_dir = "../msft_1h_intraday_data.csv"
    output_file = "msft_optimal_intraday_data.csv"

    # Read the CSV file
    df = pd.read_csv(data_dir)

    # Generate retrospective optimal trade signals
    df_with_signals = generate_retrospective_optimal_signals(df)

    # Save the DataFrame with new signals
    df_with_signals.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

    # Optional: Visualize the signals
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 7))
    plt.plot(df_with_signals['close'], label='Close Price', alpha=0.7)

    # Create scatter plot with color-coded signals
    buy_signals = df_with_signals[df_with_signals['optimal'] > 0]
    sell_signals = df_with_signals[df_with_signals['optimal'] < 0]
    neutral_signals = df_with_signals[df_with_signals['optimal'] == 0]

    plt.scatter(buy_signals.index, buy_signals['close'],
                color='green', label='Buy Signal',
                alpha=0.7, s=50)
    plt.scatter(sell_signals.index, sell_signals['close'],
                color='red', label='Sell Signal',
                alpha=0.7, s=50)
    plt.scatter(neutral_signals.index, neutral_signals['close'],
                color='gray', label='Neutral Signal',
                alpha=0.3, s=30)

    plt.title('Retrospective Optimal Trade Signals')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('retrospective_optimal_trade_signals.png')
    plt.close()


if __name__ == "__main__":
    main()