import yfinance as yf
from datetime import datetime
import numpy as np
import random

stock_symbol = 'AAPL'  # Replace with your desired stock symbol
start_date = '2023-10-23 06:00:00'  # Replace with your desired start date and time
end_date = '2023-10-27 16:00:00'    # Replace with your desired end date and time
start_date_stripped = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
end_date_stripped = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
data = yf.download(stock_symbol, start=start_date_stripped, end=end_date_stripped, interval='1m')

def calculate_moving_averages(data, sma_period, lma_period):
   
    # Calculate the short-term and long-term moving averages.
    data['SMA'] = data['Close'].rolling(window=sma_period).mean()
    data['LMA'] = data['Close'].rolling(window=lma_period).mean()

def evaluate_fitness(individual, data):
    sma_period, lma_period = individual
    calculate_moving_averages(data, sma_period, lma_period)
    
    # Assume 'Close' is the stock price column in your dataframe
    fitness = np.where(data['SMA'] > data['LMA'], data['5minutes'] - data['Close'], 0)

    return fitness,

def create_individual():
    return [random.randint(5, 50), random.randint(55, 250)]

def cx_arithmetic(ind1, ind2, alpha=0.5):
    return ([alpha * x + (1 - alpha) * y for x, y in zip(ind1, ind2)],
            [(1 - alpha) * x + alpha * y for x, y in zip(ind1, ind2)])

def mutate_individual(individual, indpb=0.05):
    if random.random() < indpb:
        individual[0] = max(5, min(50, individual[0] + random.choice([-1, 1])))
    if random.random() < indpb:
        individual[1] = max(55, min(250, individual[1] + random.choice([-1, 1])))
    return individual,

def select_tournament(population, fitness_values, tournsize):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), tournsize)
        winner = max(tournament, key=lambda i: fitness_values[i])
        selected.append(population[winner])
    return selected

def generate_next_generation(population, data, crossover_prob, mutation_prob):
    fitness_values = [evaluate_fitness(ind, data) for ind in population]

    # Select individuals for reproduction using tournament selection
    selected_parents = select_tournament(population, fitness_values, tournsize=3)

    # Perform crossover
    offspring = []
    for parent1, parent2 in zip(selected_parents[::2], selected_parents[1::2]):
        if random.random() < crossover_prob:
            child1, child2 = cx_arithmetic(parent1, parent2)
            offspring.extend([mutate_individual(child1, mutation_prob), mutate_individual(child2, mutation_prob)])
        else:
            offspring.extend([mutate_individual(parent1, mutation_prob), mutate_individual(parent2, mutation_prob)])

    # Select individuals for the next generation using elitism
    elites = [population[i] for i in np.argsort(fitness_values)[-2:]]
    next_generation = elites + offspring

    return next_generation

if __name__ == "__main__":
    # Assuming 'data' is your pandas DataFrame containing stock market data
    # ...

    # Set up the initial population
    population_size = 200
    population = [create_individual() for _ in range(population_size)]

    # Evolutionary algorithm parameters
    num_generations = 50
    crossover_prob = 0.7
    mutation_prob = 0.2

    # Evolutionary algorithm
    for _ in range(num_generations):
        population = generate_next_generation(population, data, crossover_prob, mutation_prob)

    # Retrieve the best individual from the final population
    best_individual = max(population, key=lambda ind: evaluate_fitness(id, data))

    # Print the best individual and its fitness
    print("Best Individual:", best_individual)
    print("Best Fitness:", evaluate_fitness(best_individual, data))