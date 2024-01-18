'''
Each individual in the population is only tested against one data point
'''


import yfinance as yf
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools

def download_clean_stock_data(symbols, start_date, end_date):
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
    cleaned_data['10_period_SMA'] = cleaned_data.groupby('Stock Symbol')['Close'].rolling(window=sma_window).mean().reset_index(0, drop=True)

    # Calculate the 50-period LMA
    cleaned_data['50_period_LMA'] = cleaned_data.groupby('Stock Symbol')['Close'].rolling(window=lma_window).mean().reset_index(0, drop=True)

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

def create_individual():
    # Generate three random float numbers between 0 and 1
    rand_nums = [random.uniform(0, 1) for _ in range(3)]
    
    # Calculate the sum of the random numbers
    sum_rand_nums = sum(rand_nums)
    
    # Normalize the numbers so they add up to 1.0
    normalized_nums = [num / sum_rand_nums for num in rand_nums]
    
    normalized_nums.append(0)

    return normalized_nums

def evaluate_fitness(individual, cleaned_data):


    # Indvidual index 0: SMA > LMA
    # Individual index 1: RSI < 30
    # Individual index 2: MACD strategy
    total_sum = 0.0

    random_index = random.randint(0, len(cleaned_data) - 1)
    selected_row = cleaned_data.iloc[random_index]

    if selected_row['10_period_SMA'] > selected_row['50_period_LMA']:
        total_sum += individual[0]
    if selected_row['RSI'] < 30:
        total_sum += individual[1]
    if selected_row['MACD_Line'] > selected_row['Signal_Line']:
        total_sum += individual[2]
    
    if total_sum < 0.5:
        return (individual[3])  # Fitness is 0 if the total sum is less than 0.5
    else:
        # Calculate fitness as the difference between "5minutes" value and "Close" value
        individual[3] = individual[3] + 1
        fitness = selected_row['5minutes'] - selected_row['Close']

    fitness = fitness + (individual[3])
    return fitness

def crossover(parent1, parent2, alpha):
    child1 = [0.0] * len(parent1)
    child2 = [0.0] * len(parent2)

    # Perform arithmetic blend crossover
    for i in range(len(parent1)):
        child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
        child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]

    print(f"child 1: {child1}")
    print(f"child 2: {child2}")
    
    # Normalize children's values so they add up to 1.0 
    # Normalize step may not be neccessary 
    sum_child1 = sum(child1[:-1])
    sum_child2 = sum(child2[:-1])

    normalized_child1 = [(val / sum_child1) if idx != len(child1) - 1 else val for idx, val in enumerate(child1)]
    normalized_child2 = [(val / sum_child2) if idx != len(child2) - 1 else val for idx, val in enumerate(child2)]

    return normalized_child1, normalized_child2

def mutation(individual):
    mutation_rate = 0.03  # Mutation rate of 3%

    # Perform mutation with Gaussian distribution
    for i in range(len(individual) - 1):
        if random.random() < mutation_rate:
            # Add a random value from a Gaussian distribution
            individual[i] += np.random.normal(0, 0.1)  # Adjust the parameters as needed

    # Normalize individual's values so they add up to 1.0
    sum_individual = sum(individual[:-1])
    normalized_individual = [(val / sum_individual) if idx != len(individual) - 1 else val for idx, val in enumerate(individual)]

    return normalized_individual

def create_population(population_size):
        return [create_individual() for _ in range(population_size)]

def tournament_selection(population, fitness_values, tournament_size):
    print(f"population: {population}")
    print(f"fitness values: {fitness_values}")

    selected_parents = []
    for _ in range(len(population)):
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_contestants = [fitness_values[i] for i in indices]
        winner_index = np.argmax(tournament_contestants)
        selected_parents.append(population[indices[winner_index]])
    return selected_parents

def genetic_algorithm(population_size, generations, cleaned_data):
    # Create initial population
    population = create_population(population_size)
    
    for gen in range(generations):
        print(f"generation {gen}")
        # Evaluate fitness for all individuals in the population
        fitness_values = [evaluate_fitness(individual, cleaned_data) for individual in population]
        
        # Tournament selection to select parents
        selected_parents = tournament_selection(population, fitness_values, tournament_size=3)
        
        # Create empty offspring list
        offspring = []
        
        # Crossover and Mutation
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i+1]
            
            # Perform crossover with 70% probability
            if random.random() < 0.7:
                child1, child2 = crossover(parent1, parent2, alpha=0.75)  
                
                # Perform mutation
                child1 = mutation(child1)
                child2 = mutation(child2)
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        # Replace the parents with the offspring for the next generation
        population = offspring
        
    return population

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
    # Read the CSV File
    file_path = 'cleaned_stock_data.csv'
    cleaned_data = pd.read_csv(file_path)

    # Make Technical Indicators
    calculate_moving_averages(cleaned_data, sma_window = 10, lma_window = 50)
    calculate_rsi(cleaned_data, rsi_period= 14)
    calculate_macd(cleaned_data, short_term_period = 12, long_term_period = 26, signal_period=9)
    cleaned_data['5minutes'] = cleaned_data['Close'].shift(-5)
    cleaned_data.dropna(inplace=True)  # Dropping rows with missing values
    # Handle Duplicates
    cleaned_data.drop_duplicates(inplace=True)  # Dropping duplicate rows if any
    # Reset Index
    cleaned_data.reset_index(inplace=True)
    
    population_size = 100
    generations = 50

    population = create_population(population_size)
    final_population = genetic_algorithm(population_size, generations, cleaned_data)
    sorted_population = sorted(final_population, key=lambda x: x[-1], reverse=True)
    k = 10  # Change this value to your desired number

    # Get the K best individuals from the sorted population
    k_best_individuals = sorted_population[:k]

    # Print or work with the K best individuals
    for idx, individual in enumerate(k_best_individuals, 1):
        print(f"Rank {idx} Individual: {individual}")