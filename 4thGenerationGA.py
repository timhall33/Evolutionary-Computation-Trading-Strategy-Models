'''
add candlestick indicators (incomplete)
'''

import yfinance as yf
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools


def create_individual():
    # Generate three random float numbers between 0 and 1
    rand_nums = [random.uniform(0, 1) for _ in range(3)]
    
    # Calculate the sum of the random numbers
    sum_rand_nums = sum(rand_nums)
    
    # Normalize the numbers so they add up to 1.0
    normalized_nums = [num / sum_rand_nums for num in rand_nums]
    
    return normalized_nums

def evaluate_fitness(individual, data):

    # Indvidual index 0: Moving averages strategy
    # Individual index 1: RSI strategy
    # Individual index 2: MACD strategy

    capital = 1
    bought_stock = False
    fitness = 0
    bought_stock_shares = 0
    for row in data.iterrows():
        buyTotal = 0
        sellTotal = 0
        rowData = row[1]
        if not bought_stock:
            if rowData["movingAvgBuy"] != 0:
                buyTotal += individual[0] * rowData["movingAvgBuy"]
            if rowData["movingAvgBuy"] != 0:
                buyTotal += individual[1] * rowData["rsiBuy"]
            if rowData["movingAvgBuy"] != 0:
                buyTotal += individual[2] * rowData["macdBuy"]
            if buyTotal > 0.5:
                bought_stock = True
                bought_stock_shares = capital/rowData["Close"]
        else:
            if rowData["movingAvgSell"] == 1:
                sellTotal += individual[0]
            if rowData["rsiSell"] == 1:
                sellTotal += individual[1]
            if rowData["macdSell"] == 1:
                sellTotal += individual[2]
            if sellTotal > 0.5:
                bought_stock = False
                profit = (rowData["Close"] * bought_stock_shares) - capital
                bought_stock_shares = 0
                fitness += profit
    return fitness

def crossover(parent1, parent2, alpha):
    child1 = [0.0] * len(parent1)
    child2 = [0.0] * len(parent2)

    # Perform arithmetic blend crossover
    for i in range(len(parent1)):
        child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
        child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]

    #print(f"child 1: {child1}")
    #print(f"child 2: {child2}")
    
    # Normalize children's values so they add up to 1.0 
    # Normalize step may not be neccessary 
    sum_child1 = sum(child1)
    sum_child2 = sum(child2)

    normalized_child1 = [(val / sum_child1) for val in child1]
    normalized_child2 = [(val / sum_child2) for val in child2]

    return normalized_child1, normalized_child2

def mutation(individual):
    mutation_rate = 0.03  # Mutation rate of 3%

    # Perform mutation with Gaussian distribution
    for i in range(len(individual) - 1):
        if random.random() < mutation_rate:
            # Add a random value from a Gaussian distribution
            individual[i] += np.random.normal(0, 0.1)  # Adjust the parameters as needed

    # Normalize individual's values so they add up to 1.0
    sum_individual = sum(individual)
    normalized_individual = [(val / sum_individual) for val in individual]

    return normalized_individual

def create_population(population_size):
        return [create_individual() for _ in range(population_size)]

def tournament_selection(population, fitness_values, tournament_size):
    #print(f"population: {population}")
    #print(f"fitness values: {fitness_values}")


    selected_parents = []
    for _ in range(len(population)):
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_contestants = [fitness_values[i] for i in indices]
        winner_index = np.argmax(tournament_contestants)
        selected_parents.append(population[indices[winner_index]])
    return selected_parents

def genetic_algorithm(population_size, generations, data):
    # Create initial population
    population = create_population(population_size)
    
    for gen in range(generations):
        print(f"generation {gen}")
        # Evaluate fitness for all individuals in the population
        fitness_values = [evaluate_fitness(individual, data) for individual in population]
        for ind in population:
            print(f"ind: {ind}")
        print(f"fitness values: {fitness_values}")

        # Tournament selection to select parents
        selected_parents = tournament_selection(population, fitness_values, tournament_size=3)
        
        print(f"selected parents: {selected_parents}")

        # Create empty offspring list
        offspring = []
        
        # Crossover and Mutation
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i+1]
            
            # Perform crossover with 70% probability
            if random.random() < .7:
                child1, child2 = crossover(parent1, parent2, alpha=0.75)  
                
                # Perform mutation
                child1 = mutation(child1)
                child2 = mutation(child2)
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        print(f"offspring: {offspring}")
        # Replace the parents with the offspring for the next generation
        population = offspring

    return population

if __name__ == "__main__": 
    # Read the CSV File
    file_path = 'updated_stock_data.csv'
    cleaned_data = pd.read_csv(file_path)
    
    cleaned_data.dropna(inplace=True)  # Dropping rows with missing values
    # Handle Duplicates
    cleaned_data.drop_duplicates(inplace=True)  # Dropping duplicate rows if any
    # Reset Index
    cleaned_data.reset_index(inplace=True)
    
    population_size = 10
    generations = 2

    population = create_population(population_size)
    final_population = genetic_algorithm(population_size, generations, cleaned_data)
    fitness_values = [evaluate_fitness(individual, cleaned_data) for individual in final_population]
    combined_data = list(zip(final_population, fitness_values))
    sorted_population = sorted(combined_data, key=lambda x: x[1], reverse=True)
    k = len(sorted_population)  # Change this value to desired number

    # Get the K best individuals from the sorted population
    k_best_individuals = sorted_population[:k]
    # Print or work with the K best individuals
    for idx, individual in enumerate(k_best_individuals, 1):
        print(f"Rank {idx} Individual: {individual}")