import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def calculate_moving_averages(cleaned_data, sma_window, lma_window):
    # Calculate the 10-period SMA
    cleaned_data['SMA'] = cleaned_data.groupby(['Symbol', 'Date'])['Close'].transform(lambda x: x.rolling(window=sma_window).mean())

    # Calculate the 50-period LMA
    cleaned_data['LMA'] = cleaned_data.groupby(['Symbol', 'Date'])['Close'].transform(lambda x: x.rolling(window=lma_window).mean())

def calculate_rsi(cleaned_data, rsi_period):
    cleaned_data['Price Change'] = cleaned_data.groupby(['Symbol', 'Date'])['Close'].diff()
    # Calculate the positive and negative price changes
    cleaned_data['Gain'] = cleaned_data['Price Change'].apply(lambda x: x if x > 0 else 0)
    cleaned_data['Loss'] = cleaned_data['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)

    # Calculate the average gain and average loss over the specified period (rolling mean)
    cleaned_data['Average Gain'] = cleaned_data.groupby(['Symbol', 'Date'])['Gain'].transform(lambda x: x.rolling(window=rsi_period).mean())
    cleaned_data['Average Loss'] = cleaned_data.groupby(['Symbol', 'Date'])['Loss'].transform(lambda x: x.rolling(window=rsi_period).mean())

    # Calculate the Relative Strength (RS)
    cleaned_data['RS'] = cleaned_data['Average Gain'] / cleaned_data['Average Loss']

    # Calculate the RSI using the RS
    cleaned_data['RSI'] = 100 - (100 / (1 + cleaned_data['RS']))

def calculate_macd(cleaned_data, short_term_period, long_term_period, signal_period):
    # Calculate the short-term Exponential Moving Average (EMA)
    cleaned_data[f'Short_EMA_{short_term_period}'] = cleaned_data.groupby(['Symbol', 'Date'])['Close'].transform(lambda x: x.ewm(span=short_term_period).mean())

    # Calculate the long-term Exponential Moving Average (EMA)
    cleaned_data[f'Long_EMA_{long_term_period}'] = cleaned_data.groupby(['Symbol', 'Date'])['Close'].transform(lambda x: x.ewm(span=long_term_period).mean())

    # Calculate the MACD line
    cleaned_data['MACD_Line'] = cleaned_data[f'Short_EMA_{short_term_period}'] - cleaned_data[f'Long_EMA_{long_term_period}']

    # Calculate the Signal line (typically a 9-period EMA of the MACD Line)
    cleaned_data['Signal_Line'] = cleaned_data.groupby(['Symbol', 'Date'])['MACD_Line'].transform(lambda x: x.ewm(span=signal_period).mean())


def download_stock_data(stock_symbols, start_date, end_date, start_time, end_time):
    # Create an empty dataframe to store the stock data
    stock_data = pd.DataFrame()

    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Iterate through each stock symbol
    for symbol in stock_symbols:
        # Initialize start and end dates for one week intervals
        current_date = start_date
        one_week_delta = timedelta(days=7)
        print(f"one week delta: {one_week_delta}")
        # Download data in  7-day intervals
        while current_date <= end_date:
            # Define start and end dates for one week interval
            interval_start = current_date.strftime('%Y-%m-%d')
            interval_end = min((current_date + one_week_delta), end_date).strftime('%Y-%m-%d')
            print(f"interval start: {interval_start}")
            print(f"interval end: {interval_end}")
            # Download stock data for the current symbol and date range
            try:
                stock = yf.download(symbol, start=interval_start, end=interval_end, interval='1m')
                stock['Symbol'] = symbol  # Add symbol column
                stock_data = pd.concat([stock_data, stock])  # Concatenate dataframes
            except Exception as e:
                print(f"Failed to fetch data for {symbol} from {interval_start} to {interval_end}: {e}")

            # Move to the next week interval
            current_date += one_week_delta + timedelta(days=0)  # Move one day ahead to avoid overlapping

    # Convert the index to DatetimeIndex
    stock_data.index = pd.to_datetime(stock_data.index)

    # Filter data for the specified time range
    stock_data_filtered = stock_data.between_time(start_time, end_time)

    # Save the filtered dataframe to a CSV file
    stock_data_filtered.to_csv('stock_market_data_filtered.csv')

def mark_last_20_periods(group):
        group['mustSell'] = 0  # Initialize 'mustSell' column with 0
        group.iloc[-20:, -1] = 1  # Mark the last 20 periods as 1
        return group

if __name__ == "__main__": 
    # Example usage
    stock_symbols = ['AAPL','MARA', 'AMC', 'RIOT', 'IONQ', 'AFRM', 'AMZN', 'MSFT', 'CVNA', 
                    'META', 'AMD', 'GOOG', 'MRO', 'DVN', 'PR', 'NVDA', 'ETSY', 'KLAC', 
                    'ALGN', 'TER', 'BBWI', 'NOW', 'ADBE', 'CZR', 'PYPL', 'MPWR', 'TECH',
                    'MU', 'DHI', 'SNPS', 'INTU', 'CRM']
    start_date = '2023-10-23'
    end_date = '2023-12-01'
    start_time = '09:30:00'
    end_time = '13:00:00'

    #download_stock_data(stock_symbols, start_date, end_date, start_time, end_time)

    
    file_path = 'updated_file.csv'
    data = pd.read_csv(file_path)

    data['Date'] = data['Datetime'].apply(lambda x: x.split()[0])
    
    ###################################################################################################
    # Moving Averages calculation

    calculate_moving_averages(data, sma_window = 10, lma_window = 50)

    # Calculate the movingAvgBuy column based on SMA and LMA columns
    data['movingAvgBuy'] = 0  # Initialize 'movingAvgBuy' column with 0s

    # Find rows where SMA crosses above LMA
    cross_above_indices = data[(data['SMA'] > data['LMA']) & (data['SMA'].shift(1) <= data['LMA'].shift(1))].index

    # Assign values according to specified conditions
    for idx in cross_above_indices:
        data.at[idx, 'movingAvgBuy'] = 1  # Set value to 1 where SMA crosses above LMA
        # Set values for the next 9 subsequent rows
        for i in range(1, 10):
            if idx + i < len(data):
                data.at[idx + i, 'movingAvgBuy'] = max(0, 1 - i * 0.1)

    data['movingAvgSell'] = 0  # Initialize 'movingAvgSell' column with 0s

    buy_indices = data[data['movingAvgBuy'] == 1].index

    for idx in buy_indices:
        # Set values for the next 30 periods if LMA > SMA
        for i in range(1, 31):
            if idx + i < len(data) and data.at[idx + i, 'LMA'] > data.at[idx + i, 'SMA']:
                data.at[idx + i, 'movingAvgSell'] = 1


    ###################################################################################################



    ###################################################################################################
    # RSI calculation

    calculate_rsi(data, rsi_period= 14)

    data['rsiBuy'] = 0  # Initialize 'rsiBuy' column with 0s

    rsi_indices = data[(data['RSI'] < 30) & (data['RSI'].shift(1) >= 30)].index

    # Assign values according to specified conditions
    for idx in rsi_indices:
        data.at[idx, 'rsiBuy'] = 1  # Set value to 1 where RSI crosses below 30
        # Set values for the next 9 subsequent rows
        for i in range(1, 10):
            if idx + i < len(data):
                data.at[idx + i, 'rsiBuy'] = max(0, 1 - i * 0.1)

    data['rsiSell'] = 0  # Initialize 'rsiSell' column with 0s

    buy_indices = data[data['rsiBuy'] == 1].index

    for idx in buy_indices:
        # Set values for the next 30 periods if RSI > 30
        for i in range(1, 31):
            if idx + i < len(data) and data.at[idx + i, 'RSI'] > 30:
                data.at[idx + i, 'rsiSell'] = 1

    ###################################################################################################




    ###################################################################################################
    # MACD calculation

    calculate_macd(data, short_term_period = 12, long_term_period = 26, signal_period=9)

    data['macdBuy'] = 0  # Initialize 'macdBuy' column with 0s

     # Find rows where SMA crosses above LMA
    cross_above_indices = data[(data['MACD_Line'] > data['Signal_Line']) & (data['MACD_Line'].shift(1) <= data['Signal_Line'].shift(1))].index

    # Assign values according to specified conditions
    for idx in cross_above_indices:
        data.at[idx, 'macdBuy'] = 1  # Set value to 1 where SMA crosses above LMA
        # Set values for the next 9 subsequent rows
        for i in range(1, 10):
            if idx + i < len(data):
                data.at[idx + i, 'macdBuy'] = max(0, 1 - i * 0.1)

    data['macdSell'] = 0  # Initialize 'movingAvgSell' column with 0s

    buy_indices = data[data['macdBuy'] == 1].index

    for idx in buy_indices:
        # Set values for the next 30 periods if LMA > SMA
        for i in range(1, 31):
            if idx + i < len(data) and data.at[idx + i, 'Signal_Line'] > data.at[idx + i, 'MACD_Line']:
                data.at[idx + i, 'macdSell'] = 1

    ##################################################################################################

    # Apply the function to the grouped data
    data = data.groupby(['Symbol', 'Date']).apply(mark_last_20_periods)

    data.to_csv('updated_file_2.csv', index=False)