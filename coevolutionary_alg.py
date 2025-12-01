# Knapsack - Deap
# Time
# Preset Knapsack + Deap Implementation (Contains most optimal solution)
# Comparison: Wilcoxon Ranked Sum Test


import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from knapsack import Knapsack
import numpy as np
# Fitness Function applied on all sets
# Suitable sets reproduce

# Fitness Function on each individual.
# Fit individuals are selected.
# Crossover (A pair of chromosomes) -> Mutation (each chromosome has a probibility)
# Calculate new Fitness for each individual

# Create an instance of a Knapsack problem
p_file = "Knapsack_Problems/problemInstances/n_400_c_1000000_g_2_f_0.1_eps_0.0001_s_100/test.in"
opt_file = "Knapsack_Problems/optima.csv"
knapsack = Knapsack(p_file, opt_file)

# Defining Problem Constraints
ITEMS_LENGTH = len(knapsack)
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 8
# Random Seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# Create the binary list operator
toolbox = base.Toolbox()
toolbox.register('zeroOrOne', random.randint, 0, 1)

# Create the Fitness Class
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Weight of 1 means we want to MAXIMIZE the fitness
# Create the individual class
creator.create('Individual', list, fitness=creator.FitnessMax)
# Register the individualCreator operator
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ITEMS_LENGTH)
# Register the popuilationCreator operator
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)

# Define a function to evaluate the total value of the selected items
def evaluate(individual):
    return knapsack.getTotalValue(individual), # return a tuple

# Register the countOnes function
toolbox.register("evaluate", evaluate)

# Create the genetic operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)

# Create the statistics object
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Register the statistics object
stats.register("max", np.max)
stats.register("avg", np.mean)

# Create the hall of fame object
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Define the algorithm
population, logbook = algorithms.eaSimple(toolbox.populationCreator(n=POPULATION_SIZE), toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

# Print the best ever individual
print("Best Individual: ", hof.items[0])
print("Fitness:", knapsack.getTotalValue(hof.items[0]))
knapsack.printItems(hof.items[0])

# Return the max and mean values
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

# Plot the max and mean vals
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.legend(('Max', 'Mean'), loc='lower right')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Mean Fitness over Generations')
plt.show()
