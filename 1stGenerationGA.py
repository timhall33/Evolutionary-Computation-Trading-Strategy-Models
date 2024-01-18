import yfinance as yf
from datetime import datetime
import pandas as pd
import random
from deap import base, creator, tools, algorithms


stock_symbol = 'AAPL'  # Replace with your desired stock symbol
start_date = '2023-10-23 09:30:00'  # Replace with your desired start date and time
end_date = '2023-10-27 16:00:00'    # Replace with your desired end date and time
start_date_stripped = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
end_date_stripped = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
data = yf.download(stock_symbol, start=start_date_stripped, end=end_date_stripped, interval='1m')
df = pd.DataFrame(data)
df['5minutes'] = df['Close'].shift(-5)


def calculate_moving_averages(df, sma_period, lma_period, row_index):
   
    # Calculate the short-term and long-term moving averages.
    SMA = df["Close"].iloc[row_index : row_index + sma_period].mean()
    LMA = df["Close"].iloc[row_index : row_index + lma_period].mean()
    return SMA, LMA

def evaluate_fitness(individual, df):


    # add a parameter where a random row is picked each generation (not for each individual)
    # try it again but pick a different row for each indiviual in each generation

    row_index = random.randint(250, len(df) - 6)
    selected_row = df.iloc[row_index]
    sma_period, lma_period = individual
    SMA, LMA = calculate_moving_averages(df, sma_period, lma_period, row_index)
    print(f"individual: {individual}")
    print(f"LMA: {LMA}")
    print(f"SMA: {SMA}")

    if SMA > LMA:
        fitness = selected_row['5minutes'] - selected_row['Close']
    else:
        fitness = 0
    
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

if __name__ == "__main__":
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, creator.Individual, toolbox.individual)

    toolbox.register("evaluate", evaluate_fitness, df=df)
    toolbox.register("mate", cx_arithmetic)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Set up the population
    population_size = 200
    population = toolbox.population(n=population_size)
    
    # Evolutionary algorithm parameters
    num_generations = 1000
    crossover_prob = 0.7
    mutation_prob = 0.2
    
    # Evolutionary algorithm
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=crossover_prob,
                              mutpb=mutation_prob, ngen=num_generations, stats=None, halloffame=None, verbose=True)
    
    # Retrieve the best individual from the final population
    best_individual = tools.selBest(population, k=1)[0]
    
    # Print the best individual and its fitness
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_individual.fitness.values[0])