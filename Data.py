import pandas as pd
import yfinance as yf
import numpy as np


def download_stock_data(symbols, start_date, end_date):
    data = {}
    
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1m')
        stock_data['Stock Symbol'] = symbol  # Adding stock symbol as a column
        data[symbol] = stock_data
    
    # Concatenating data for all stocks into a single DataFrame
    df = pd.concat(data.values(), axis=0)
    
    # Handle Missing Values
    df.dropna(inplace=True)  # Dropping rows with missing values

    # Handle Duplicates
    df.drop_duplicates(inplace=True)  # Dropping duplicate rows if any

    # Reset Index
    df.reset_index(inplace=True)
    
    # Save the cleaned data to a CSV file
    df.to_csv('cleaned_stock_data.csv_2', index=False)
    
    return df


def calculate_moving_averages(cleaned_data, sma_window, lma_window):
    # Calculate the 10-period SMA
    cleaned_data['SMA'] = cleaned_data.groupby('Stock Symbol')['Close'].rolling(window=sma_window).mean().reset_index(0, drop=True)

    # Calculate the 50-period LMA
    cleaned_data['LMA'] = cleaned_data.groupby('Stock Symbol')['Close'].rolling(window=lma_window).mean().reset_index(0, drop=True)

def calculate_rsi(cleaned_data, rsi_period):
    cleaned_data['Price Change'] = cleaned_data.groupby('Stock Symbol')['Close'].diff()
    # Calculate the positive and negative price changes
    cleaned_data['Gain'] = cleaned_data['Price Change'].apply(lambda x: x if x > 0 else 0)
    cleaned_data['Loss'] = cleaned_data['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)

    # Calculate the average gain and average loss over the specified period (rolling mean)
    cleaned_data['Average Gain'] = cleaned_data.groupby('Stock Symbol')['Gain'].rolling(window=rsi_period).mean().reset_index(0, drop=True)
    cleaned_data['Average Loss'] = cleaned_data.groupby('Stock Symbol')['Loss'].rolling(window=rsi_period).mean().reset_index(0, drop=True)

    # Calculate the Relative Strength (RS)
    cleaned_data['RS'] = cleaned_data['Average Gain'] / cleaned_data['Average Loss']

    # Calculate the RSI using the RS
    cleaned_data['RSI'] = 100 - (100 / (1 + cleaned_data['RS']))

def calculate_macd(cleaned_data, short_term_period, long_term_period, signal_period):
    # Calculate the short-term Exponential Moving Average (EMA)
    cleaned_data[f'Short_EMA_{short_term_period}'] = cleaned_data.groupby('Stock Symbol')['Close'].transform(lambda x: x.ewm(span=short_term_period).mean())

    # Calculate the long-term Exponential Moving Average (EMA)
    cleaned_data[f'Long_EMA_{long_term_period}'] = cleaned_data.groupby('Stock Symbol')['Close'].transform(lambda x: x.ewm(span=long_term_period).mean())

    # Calculate the MACD line
    cleaned_data['MACD_Line'] = cleaned_data[f'Short_EMA_{short_term_period}'] - cleaned_data[f'Long_EMA_{long_term_period}']

    # Calculate the Signal line (typically a 9-period EMA of the MACD Line)
    cleaned_data['Signal_Line'] = cleaned_data.groupby('Stock Symbol')['MACD_Line'].transform(lambda x: x.ewm(span=signal_period).mean())
'''
def identify_hammer_candlestick(data):
    # Calculate relevant candlestick characteristics
    data['Real Body'] = abs(data['Open'] - data['Close'])
    data['Upper Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
    data['Lower Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']

    # Define conditions for a hammer candlestick
    is_downtrend = False
    hammer_detected = []

    for i in range(1, len(data)):
        threshold = -0.05

        if i > 2:
            recent_closes = [data['Close'][i - j] for j in range(3)]
    
            # Calculate the slope of the closing prices
            close_slope = np.polyfit(range(3), recent_closes, 1)[0]
            
            # Check if the slope is more negative than the threshold
            if close_slope < threshold:
                is_downtrend = True
            else:
                is_downtrend = False
        
        # Check for hammer pattern
        if is_downtrend and data['Lower Shadow'][i] >= 2 * data['Real Body'][i] and data['Upper Shadow'][i] < data['Real Body'][i]:
            hammer_detected.append(1)
        else:
            hammer_detected.append(0)

        # Reset downtrend flag
        is_downtrend = False

    # Add the hammer detection column to the dataframe
    data['Hammer'] = [0] + hammer_detected

    return data
'''
if __name__ == "__main__": 

    '''
    # Define the stock symbols, start date, and end date
    stock_symbols = ['AAPL','MARA', 'AMC', 'RIOT', 'IONQ', 'AFRM', 'AMZN', 'MSFT', 'CVNA', 
                    'META', 'AMD', 'GOOG', 'MRO', 'DVN', 'PR', 'NVDA', 'ETSY', 'KLAC', 
                    'ALGN', 'TER', 'BBWI', 'NOW', 'ADBE', 'CZR', 'PYPL', 'MPWR', 'TECH',
                     'MU', 'DHI', 'SNPS', 'INTU', 'CRM']
    start_date = '2023-11-06'
    end_date = '2023-11-10'

    # Run the function to download, clean, and save the data
    cleaned_data = download_clean_stock_data(stock_symbols, start_date, end_date)
    '''

    # Load the CSV file into a pandas DataFrame
    file_path = 'cleaned_stock_data.csv'
    data = pd.read_csv(file_path)

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


    data.to_csv('updated_file.csv', index=False)

