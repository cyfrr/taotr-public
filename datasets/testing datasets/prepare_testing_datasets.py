import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Iterate over all files in the directory
for filename in os.listdir(script_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(script_directory, filename)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Rename the 'timestamp' column to 'date' if it exists
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'date'}, inplace=True)
            df.index = df['date']

        df['optimal'] = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # Resample the data from 1 minute to 5 minutes
        df.index = pd.to_datetime(df.index)  # Ensure the 'date' column is in datetime format
        df_5min = df.resample('1h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

        # Save the resampled DataFrame as a new CSV file with '5min' in the name
        resampled_filename = filename.replace('1min', '1h')
        df_5min.to_csv(os.path.join(script_directory, resampled_filename))

        # Reset the index for plotting
        df_5min.reset_index(inplace=True)

        # Plotting the data
        plt.figure()
        df.plot(x='date', y='close', title=filename)
        plt.xlabel('date')
        plt.ylabel('value')

        # Close the plot to free up memory
        plt.show()
