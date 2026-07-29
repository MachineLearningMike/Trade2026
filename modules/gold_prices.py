import yfinance as yf
import pandas as pd
from datetime import datetime

# --- Configuration ---
# Ticker symbol for Gold Futures (GC=F) is commonly used for continuous price data[citation:6].
# You can also use "XAUUSD=X" for the spot price, though it may have slightly less historical data.
TICKER = "GC=F"
START_DATE = "2024-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d') # Collects data up to today

# --- Download Data ---
print(f"Downloading gold price data for {TICKER} from {START_DATE} to {END_DATE}...")
try:
    # The 'auto_adjust' parameter consolidates price data for a cleaner view.
    gold_data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
except Exception as e:
    print(f"An error occurred during download: {e}")
    exit()

# --- Data Cleaning ---
if gold_data.empty:
    print("No data was downloaded. Please check the ticker or date range.")
    exit()

# Drop any rows with missing data to ensure a clean dataset
gold_data.dropna(inplace=True)

# The 'auto_adjust' parameter often returns a 'Close' column; let's rename it for clarity.
# It might also return 'Volume'.
if 'Close' in gold_data.columns:
    gold_data.rename(columns={'Close': 'Price'}, inplace=True)

# --- Display and Save ---
print(f"Successfully downloaded {len(gold_data)} days of data.")
print("\nFirst 5 rows of the data:")
print(gold_data.head())

print("\nLast 5 rows of the data:")
print(gold_data.tail())

# Save the data to a CSV file for future use
csv_filename = f"gold_prices_{START_DATE}_to_{END_DATE}.csv"
gold_data.to_csv(csv_filename)
print(f"\nData saved to {csv_filename}")