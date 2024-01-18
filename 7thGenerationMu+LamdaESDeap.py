import pandas as pd
import random
import numpy as np
from deap import base, creator, tools, algorithms
import json
import matplotlib.pyplot as plt
import time

def create_individual():
    # Generate 6 random floats
    floats = [random.uniform(0, 1) for _ in range(26)]

    # Normalize the first 3 floats to sum up to 1
    sum_first_half = sum(floats[:13])
    floats[:13] = [val / sum_first_half for val in floats[:13]]

    # Normalize the last 3 floats to sum up to 1
    sum_second_half = sum(floats[13:])
    floats[13:] = [val / sum_second_half for val in floats[13:]]

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
                    if rowData["morningStar"] > 0:
                        buyTotal += individual[3] * (rowData["morningStar"] / 100)
                    if rowData["hammer"] > 0:
                        buyTotal += individual[4] * (rowData["hammer"] / 100)
                    if rowData["piercing"] > 0:
                        buyTotal += individual[5] * (rowData["piercing"] / 100)
                    if rowData["invertedHammer"] > 0:
                        buyTotal += individual[6] * (rowData["invertedHammer"] / 100)
                    if rowData["threeWhiteSoldiers"] > 0:
                        buyTotal += individual[7] * (rowData["threeWhiteSoldiers"] / 100)
                    if rowData["engulfing"] > 0:
                        buyTotal += individual[8] * (rowData["engulfing"] / 100)
                    if rowData["hamari"] > 0:
                        buyTotal += individual[9] * (rowData["hamari"] / 100)
                    if rowData["beltHold"] > 0:
                        buyTotal += individual[10] * (rowData["beltHold"] / 100)
                    if rowData["threeInsideUp"] > 0:
                        buyTotal += individual[11] * (rowData["threeInsideUp"] / 100)
                    if rowData["kicker"] > 0:
                        buyTotal += individual[12] * (rowData["kicker"] / 100)
                    if buyTotal > 0.1 and rowData["mustSell"] != 1:
                        bought_stock = True
                        bought_stock_shares = capital/rowData["Close"]
                else:
                    periodsHeld += 1
                    if rowData["movingAvgSell"] == 1:
                        sellTotal += individual[13]
                    if rowData["rsiSell"] == 1:
                        sellTotal += individual[14]
                    if rowData["macdSell"] == 1:
                        sellTotal += individual[15]
                    if rowData["engulfing"] < 0:
                        sellTotal += individual[16] * (rowData["engulfing"] / -100)
                    if rowData["hamari"] < 0:
                        sellTotal += individual[17] * (rowData["hamari"] / -100)
                    if rowData["beltHold"] < 0:
                        sellTotal += individual[18] * (rowData["beltHold"] / -100)
                    if rowData["threeInsideUp"] < 0:
                        sellTotal += individual[19] * (rowData["threeInsideUp"] / -100)
                    if rowData["kicker"] < 0:
                        sellTotal += individual[20] * (rowData["kicker"] / -100)
                    if rowData["shootingStar"] < 0:
                        sellTotal += individual[21] * (rowData["shootingStar"] / -100)
                    if rowData["darkCloudCover"] < 0:
                        sellTotal += individual[22] * (rowData["darkCloudCover"] / -100)
                    if rowData["eveningStarList"] < 0:
                        sellTotal += individual[23] * (rowData["eveningStarList"] / -100)
                    if rowData["hangingMan"] < 0:
                        sellTotal += individual[24] * (rowData["hangingMan"] / -100)
                    if rowData["threeBlackCrows"] < 0:
                        sellTotal += individual[25] * (rowData["threeBlackCrows"] / -100)
                    if sellTotal > 0.1 or periodsHeld >= 20 or rowData["mustSell"] == 1:
                        bought_stock = False
                        profit = (rowData["Close"] * bought_stock_shares) - capital
                        bought_stock_shares = 0
                        fitness += profit
                        periodsHeld = 0
    return (fitness,)

def es_mutation(individual, mutation_step_size):
    mutation_rate = 0.05  # Adjust mutation rate if needed

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            random_num = np.random.normal(0, mutation_step_size)
            max_value = individual[i] + random_num

            if max_value >= 0:
                individual[i] += random_num 

    # Normalize mutated individual's values so they add up to 1.0
    sum_first_half = sum(individual[:13])
    sum_second_half = sum(individual[13:])
    individual[:13] = [val / sum_first_half for val in individual[:13]]
    individual[13:] = [val / sum_second_half for val in individual[13:]]

def evolution_strategy(data):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialize DEAP toolbox
    toolbox = base.Toolbox()

    # Register the custom functions
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, data=data)
    toolbox.register("mutate", es_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)


    # ES parameters
    population_size = 30
    num_generations = 15
    initial_mutation_step_size = 0.08  # Initial mutation step size
    max_step_increase = 1.5  # Maximum increase factor for step size
    max_step_decrease = 0.5  # Maximum decrease factor for step size

    mutation_step_size = initial_mutation_step_size

    # Initialize lists to store generational statistics
    min_fitness_list = []
    max_fitness_list = []
    avg_fitness_list = []

    # Initialization
    population = toolbox.population(n=population_size)

    # Evaluate the entire population
    fitness_values = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitness_values):
        ind.fitness.values = fit

    sTime = time.time()

    for generation in range(num_generations):
        start_time = time.time()  # Record the start time

        # Select the next generation's individuals using tournament selection
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Calculate mean fitness of individuals
        mean_best_fitness = np.mean([ind.fitness.values[0] for ind in population])

        # Adjust mutation step size based on mean fitness change
        if generation > 0:
            fitness_change_ratio = mean_best_fitness / previous_mean_best_fitness
            if fitness_change_ratio > 1.0:
                mutation_step_size *= min(max_step_increase, fitness_change_ratio)
            else:
                mutation_step_size *= max(max_step_decrease, fitness_change_ratio)

        # Apply boundarie to mutation step size
        mutation_step_size = max(0.01, min(mutation_step_size, 0.5))  # Example boundaries

        previous_mean_best_fitness = mean_best_fitness  # Update for next iteration

        for mutant in offspring:
            toolbox.mutate(mutant, mutation_step_size=mutation_step_size)
            del mutant.fitness.values

        # Evaluate the offspring individuals with invalid fitness values
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitness_values = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitness_values):
            ind.fitness.values = fit

        # Replace the current population with the offspring
        combined_population = population + offspring
        # Replace the current population with the offspring
        population[:] = toolbox.select(combined_population, len(population))

        # Gather and print the population's statistics
        fits = [ind.fitness.values[0] for ind in population]
        min_fitness_list.append(min(fits))
        max_fitness_list.append(max(fits))
        avg_fitness_list.append(sum(fits) / len(population))
        end_time = time.time()  # Record the end time
        generation_time = end_time - start_time  # Calculate the time taken for the generation
        print(f"-- Generation {generation + 1} --")
        print(f"  Min fitness: {min(fits)}")
        print(f"  Max fitness: {max(fits)}")
        print(f"  Avg fitness: {sum(fits) / len(population)}")
        print(f"  Time taken for generation: {generation_time:.4f} seconds")

    best_individuals = tools.selBest(population, len(population))
    best_individuals_sorted = sorted(best_individuals, key=lambda x: x.fitness.values[0], reverse=True)
    for idx, ind in enumerate(best_individuals_sorted, start=1):
        print(f"Individual {idx}: {ind} - Fitness: {ind.fitness.values[0]}")


    eTime = time.time()
    totalTime = eTime - sTime
    print(f"  Time taken for all generations: {totalTime:.4f} seconds")

    generations = range(1, num_generations + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, min_fitness_list, label='Min Fitness', marker='o')
    plt.plot(generations, max_fitness_list, label='Max Fitness', marker='o')
    plt.plot(generations, avg_fitness_list, label='Avg Fitness', marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

    serializable_individuals = [
    {
        'individual': ind,
        'fitness': ind.fitness.values[0]
    }
    for ind in best_individuals_sorted]
    output_file_json = 'saved_individuals_5.json'
    with open(output_file_json, 'w') as json_file:
        json.dump(serializable_individuals, json_file, indent=4)

if __name__ == "__main__":
    file_path = 'updated_file_3.csv'
    data = pd.read_csv(file_path)
    evolution_strategy(data)
