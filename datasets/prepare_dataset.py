import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_outlier_detection(cleaned_data, original_data, symbol='MSFT'):
    """
    Plot the original and cleaned data to visualize the removed jumps
    """
    plt.figure(figsize=(15, 8))

    plt.plot(original_data.index, original_data['close'], 'r.',
             label='Original Data', alpha=0.5, markersize=2)
    plt.plot(cleaned_data.index, cleaned_data['close'], 'b-',
             label='Cleaned Data', linewidth=1)

    plt.title(f'{symbol} 1-Hour Intraday Close Prices - Jump Detection')
    plt.xlabel('Time')
    plt.ylabel('Close Price USD')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def remove_extreme_jumps(df, jump_threshold=0.1):
    """
    Remove extreme price jumps that are clearly errors
    """
    cleaned_data = df.copy()
    price_columns = ['open', 'high', 'low', 'close']

    # Get the day for each timestamp
    cleaned_data['date'] = cleaned_data.index.date

    for column in price_columns:
        # Calculate price changes
        price_changes = cleaned_data[column].pct_change()

        # Get previous day's last price and next day's first price for each point
        prev_day_prices = {}
        next_day_prices = {}

        for date in cleaned_data['date'].unique():
            day_mask = cleaned_data['date'] == date
            if day_mask.any():
                day_data = cleaned_data[day_mask][column]
                prev_day_prices[date] = day_data.iloc[0]
                next_day_prices[date] = day_data.iloc[-1]

        # Identify suspicious jumps
        suspicious_jumps = []

        for i in range(1, len(cleaned_data) - 1):
            current_price = cleaned_data[column].iloc[i]
            current_date = cleaned_data['date'].iloc[i]

            # Get previous and next day prices for comparison
            prev_day = cleaned_data['date'].iloc[i - 1]
            next_day = cleaned_data['date'].iloc[i + 1]

            prev_price = prev_day_prices.get(prev_day)
            next_price = next_day_prices.get(next_day)

            if prev_price is not None and next_price is not None:
                # Calculate percentage differences
                pct_diff_prev = abs(current_price - prev_price) / prev_price
                pct_diff_next = abs(current_price - next_price) / next_price

                # If price is significantly different from both previous and next day
                if pct_diff_prev > jump_threshold and pct_diff_next > jump_threshold:
                    suspicious_jumps.append(i)

        # Interpolate suspicious jumps
        if suspicious_jumps:
            print(f"Found {len(suspicious_jumps)} extreme jumps in {column}")
            for idx in suspicious_jumps:
                cleaned_data.iloc[idx, cleaned_data.columns.get_loc(column)] = np.nan
                cleaned_data[column] = cleaned_data[column].interpolate(method='linear')

    # Remove the temporary date column
    cleaned_data = cleaned_data.drop('date', axis=1)

    return cleaned_data


def main():
    # Directory containing the yearly datasets
    data_dir = "1s yearly datasets"
    # Output file name
    output_file = "msft_1h_intraday_data.csv"
    output_file_cleaned = "msft_1h_intraday_data.csv"

    # Initialize an empty list to hold dataframes
    dfs = []

    # Define the column names based on the CSV format
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Loop through each year from 2017 to 2024
    for year in range(2017, 2024 + 1):
        file_path = os.path.join(data_dir, f"msft_intraday_data_{year}.csv")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing data for {year}...")

        # Read the CSV file with the correct column names and skip the first row if it contains headers
        df_1s = pd.read_csv(file_path, names=columns,
                            header=0 if 'date' in open(file_path).readline().lower() else None)

        try:
            # Convert 'date' column to datetime
            df_1s['date'] = pd.to_datetime(df_1s['date'], errors='coerce')

            # Set the date column as the index
            df_1s.set_index('date', inplace=True)

            # Convert price columns to numeric, removing any non-numeric characters
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df_1s[col] = pd.to_numeric(df_1s[col].astype(str).str.replace('[^\d.]', ''), errors='coerce')

            # Convert volume to numeric
            df_1s['volume'] = pd.to_numeric(df_1s['volume'], errors='coerce')

            # Resample the data to 1 hour intervals
            df_1h = df_1s.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Append to list of dataframes
            dfs.append(df_1h)
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # If no valid dataframes were found, exit the script
    if not dfs:
        print("No valid dataframes found. Exiting.")
        return

    # Concatenate all yearly dataframes
    result_df = pd.concat(dfs)

    # Sort data chronologically
    result_df = result_df.sort_index()

    # Save the original concatenated dataframe
    result_df.to_csv(output_file)
    print(f"Original data saved to {output_file}")


if __name__ == "__main__":
    main()
