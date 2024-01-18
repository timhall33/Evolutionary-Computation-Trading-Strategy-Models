import yfinance as yf
from datetime import datetime


stock_symbol = 'AAPL'  # Replace with your desired stock symbol
start_date = '2023-10-23 06:00:00'  # Replace with your desired start date and time
end_date = '2023-10-27 16:00:00'    # Replace with your desired end date and time
start_date_stripped = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
end_date_stripped = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
data = yf.download(stock_symbol, start=start_date_stripped, end=end_date_stripped, interval='1m')
print(data.head(10))
data["5minutes"] = data["Close"].shift(-5)
print(data.head(10))
tickers = ['MARA', 'AMC', 'RIOT', 'IONQ', 'AFRM', 'AMZN', 'MSFT', 'CVNA', 'META', 'AMD', 'GOOG', 'MRO', 'DVN', 'PR']

train = data.iloc[:-100]
test = data.iloc[-100:]

import random

# Define the population size
population_size = 50

# Define the length of an individual (chromosome)
individual_length = 4  # Represents open, high, low, and close thresholds

# Define the range for threshold values (e.g., between 0 and 1)
threshold_range = (0, 1)

# Create an initial population
def create_individual():
    return [random.uniform(threshold_range[0], threshold_range[1]) for _ in range(individual_length)]

def create_population():
    return [create_individual() for _ in range(population_size)]

# Example of creating an initial population
population = create_population()
print("Initial Population:")
for ind in population:
    print(ind)
