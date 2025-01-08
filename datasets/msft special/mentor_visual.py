import pandas as pd

# Provide the file path to your dataset
file_path = 'msft_mentor.csv'

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(file_path)

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])


# Create a function to fill in the missing data
def fill_missing_data(df):
    # Get the minimum and maximum dates in the DataFrame
    start_date = pd.Timestamp('2017-01-01')
    end_date = pd.Timestamp('2024-01-01')

    # Create a new DataFrame with all the dates and hours
    all_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    new_df = pd.DataFrame({'date': all_dates})

    # Merge the original DataFrame with the new one, filling in missing data
    merged_df = pd.merge(new_df, df, on='date', how='left').fillna(method='ffill')

    return merged_df


# Call the function to fill in the missing data
updated_df = fill_missing_data(df)

# Provide the file path to your stock data
stock_data_path = "../msft_1h_intraday_data.csv"

# Load the stock data into a Pandas DataFrame
stock_df = pd.read_csv(stock_data_path)

# Ensure that both 'date' columns are in datetime format
stock_df['date'] = pd.to_datetime(stock_df['date'])
updated_df['date'] = pd.to_datetime(updated_df['date'])

# Remove rows in stock_df with dates past 2024-01-01
cutoff_date = pd.Timestamp('2024-01-01')
stock_df = stock_df[stock_df['date'] <= cutoff_date]

# Merge the updated DataFrame with the stock data based on the 'date' column
merged_with_stock_df = pd.merge(stock_df, updated_df[['date', 'mentor']], on='date', how='left')

# Save the final merged DataFrame to a CSV file
merged_with_stock_df.to_csv('msft_mentor_intraday_data.csv', index=False)

print('Dataset updated with missing data filled in.')
