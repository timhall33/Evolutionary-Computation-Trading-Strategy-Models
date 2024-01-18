import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import json

'''
Complete Deap Algorithm with 3 trading strategies (Moving Averages, MACD, RSI)
'''

def create_individual():
    # Generate 6 random floats
    floats = [random.uniform(0, 1) for _ in range(6)]

    # Normalize the first 3 floats to sum up to 1
    sum_first_half = sum(floats[:3])
    floats[:3] = [val / sum_first_half for val in floats[:3]]

    # Normalize the last 3 floats to sum up to 1
    sum_second_half = sum(floats[3:])
    floats[3:] = [val / sum_second_half for val in floats[3:]]

    return floats

def evaluate(individual, data):
    # Indvidual index 0: Moving averages strategy
    # Individual index 1: RSI strategy
    # Individual index 2: MACD strategy

    capital = 1
    bought_stock = False
    fitness = 0
    bought_stock_shares = 0
    periodsHeld = 0

    symbols = data['Symbol'].unique()

    # Loop through each symbol
    for symbol in symbols:
        symbol_data = data[data['Symbol'] == symbol]  # Filter data for the current symbol
        symbol_dates = symbol_data['Date'].unique()  # Get unique dates for the symbol
        
        # Calculate the number of days that represent 80% of the data
        eighty_percent = int(len(symbol_dates) * 0.8)
        
        # Extract the first 80% of the days for the symbol
        dates_80_percent = symbol_dates[:eighty_percent]
        
        # Loop through the data for the first 80% of the days
        for date in dates_80_percent:
            subset_data = symbol_data[symbol_data['Date'] == date]  # Filter data for the current date
            for row in subset_data.iterrows():
                buyTotal = 0
                sellTotal = 0
                rowData = row[1]
                if not bought_stock:
                    if rowData["movingAvgBuy"] != 0:
                        buyTotal += individual[0] * rowData["movingAvgBuy"]
                    if rowData["rsiBuy"] != 0:
                        buyTotal += individual[1] * rowData["rsiBuy"]
                    if rowData["macdBuy"] != 0:
                        buyTotal += individual[2] * rowData["macdBuy"]
                    if buyTotal > 0.5 and rowData["mustSell"] != 1:
                        bought_stock = True
                        bought_stock_shares = capital/rowData["Close"]
                else:
                    periodsHeld += 1
                    if rowData["movingAvgSell"] == 1:
                        sellTotal += individual[3]
                    if rowData["rsiSell"] == 1:
                        sellTotal += individual[4]
                    if rowData["macdSell"] == 1:
                        sellTotal += individual[5]
                    if sellTotal > 0.5 or periodsHeld >= 20 or rowData["mustSell"] == 1:
                        bought_stock = False
                        profit = (rowData["Close"] * bought_stock_shares) - capital
                        bought_stock_shares = 0
                        fitness += profit
                        periodsHeld = 0
    return (fitness,)

def crossover(parent1, parent2):
    alpha = 0.75
    child1 = [0.0] * len(parent1)
    child2 = [0.0] * len(parent2)

    # Perform arithmetic blend crossover
    for i in range(len(parent1)):
        child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
        child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]

    # Normalize children's values so they add up to 1.0 
    # Normalize step may not be neccessary 

    sum_child1_first_half = sum(child1[:3])
    sum_child1_second_half = sum(child1[3:])
    sum_child2_first_half = sum(child2[:3])
    sum_child2_second_half = sum(child2[3:])

    child1[:3] = [(val / sum_child1_first_half) for val in child1[:3]]
    child1[3:] = [(val / sum_child1_second_half) for val in child1[3:]]
    child2[:3] = [(val / sum_child2_first_half) for val in child2[:3]]
    child2[3:] = [(val / sum_child2_second_half) for val in child2[3:]]

    return child1, child2

def mutation(individual):
    mutation_rate = 1  # Mutation rate of 3%

    # Perform mutation with Gaussian distribution
    for i in range(len(individual) - 1):
        if random.random() < mutation_rate:
            randomNum = np.random.normal(0, 0.1)
            max_value = individual[i] + randomNum
        
            # Add random value only if it won't make individual[i] go below 0
            if max_value >= 0:
                individual[i] += randomNum  

    # Normalize individual's values so they add up to 1.0
    sum_individual_first_half = sum(individual[:3])
    sum_individual_second_half = sum(individual[3:])
    individual[:3] = [(val / sum_individual_first_half) for val in individual[:3]]
    individual[3:] = [(val / sum_individual_second_half) for val in individual[3:]]

def genetic_algorithm(data):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialize DEAP toolbox
    toolbox = base.Toolbox()

    # Register the custom functions
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, data=data)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutation)
    toolbox.register("select", tools.selTournament, tournsize=2)

    # Genetic Algorithm parameters
    population_size = 4
    num_generations = 2
    crossover_prob = 1
    mutation_prob = 1

    # Create initial population
    population = toolbox.population(n=population_size)

    # Evaluate the entire population
    fitness_values = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitness_values):
        ind.fitness.values = fit

    # Begin the evolution process
    for generation in range(num_generations):

        # Select the next generation's individuals using tournament selection
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation to the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the offspring individuals with invalid fitness values
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitness_values = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitness_values):
            ind.fitness.values = fit

        # Replace the current population with the offspring
        population[:] = offspring

         # Gather and print the population's statistics
        fits = [ind.fitness.values[0] for ind in population]
        print(f"-- Generation {generation + 1} --")
        print(f"  Min fitness: {min(fits)}")
        print(f"  Max fitness: {max(fits)}")
        print(f"  Avg fitness: {sum(fits) / len(population)}")

    best_individuals = tools.selBest(population, len(population))
    best_individuals_sorted = sorted(best_individuals, key=lambda x: x.fitness.values[0], reverse=True)
    for idx, ind in enumerate(best_individuals_sorted, start=1):
        print(f"Individual {idx}: {ind} - Fitness: {ind.fitness.values[0]}")


    serializable_individuals = [
    {
        'individual': ind,
        'fitness': ind.fitness.values[0]
    }
    for ind in best_individuals_sorted]
    output_file_json = 'saved_individual.json'
    with open(output_file_json, 'w') as json_file:
        json.dump(serializable_individuals, json_file, indent=4)

if __name__ == "__main__": 
    file_path = 'updated_file_2.csv'
    data = pd.read_csv(file_path)

    '''
    # Extract only the date part from the 'Date' column
    data['Date'] = data['Datetime'].apply(lambda x: x.split()[0])

    # Group the data by 'Ticker' and 'Date'
    grouped_data = data.groupby(['Symbol', 'Date'])

    # Create an empty array to store dataframes
    dataframes_array = []

    # Iterate through each group and store the dataframes in the array
    for (ticker, date), group in grouped_data:
        # Create a separate dataframe for each group
        df = pd.DataFrame(group)
        
        # Append the dataframe to the array
        dataframes_array.append(df)
    '''
    genetic_algorithm(data)
    
