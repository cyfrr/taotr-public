import yfinance as yf
from datetime import datetime, timedelta

# Define the stock symbol
ticker_symbol = 'UNH'  # UnitedHealth Group's ticker symbol

# Calculate the date range for the past year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Download historical data
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Select only the necessary columns and rename them to lowercase
stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
stock_data.columns = ['open', 'high', 'low', 'close', 'volume']

# Save the data to a CSV file
stock_data.to_csv('UHC_historical_data.csv')

# Print out some basic information
print(f"Downloaded historical data for {ticker_symbol}")
print(f"Date range: {start_date.date()} to {end_date.date()}")
print(f"Total number of trading days: {len(stock_data)}")
print("\nFirst few rows:")
print(stock_data.head())

print("\nLast few rows:")
print(stock_data.tail())

# Basic statistics
print("\nBasic statistics:")
print(stock_data['close'].describe())
