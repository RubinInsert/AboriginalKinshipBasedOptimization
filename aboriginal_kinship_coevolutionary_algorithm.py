# KNAPSACK PROBLEMS: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html
# KEY FEATURES:
# 8 Groups
# Uses cxUniform Crossover
# Uses mutFlipBit Mutation
# Individuals are initialized through the 'init_hybrid_population' function which uses a greedy algorithm on a sorted v/w ratio list.
# - Without this individuals are too far from any probable options for the GA to make any progress.
# Elitism is used from the previous generation (the parents).
# Children are exposed to a repair_operator prior to evaluation.
# - This works "backwards" placing all items in a v/w ratio list and removing the lowest items until the individual is viable.
# TODO:
# Implement HOF. Different to Eliteism (Global effect)
# Change from limiting generations, to limiting algorithm run-time. (Continue tracking generations for additional statistics however)
# - What should be considered the start of the algorithm? I.e. is the initialization of population included?
# Possibly update the generic evolutionary algorithm to include the above key features for an additional benchmark
# Generic Co-evolutionary algorithm
# Questions:
# - What Co-evolutionary algorithm would be best to benchmark against? E.g. competitive or cooperative? I presume competitive?
# - How should the generic co-evolutionary algorithm function? should the children just be placed in random groups? are parents chosen from their same group?
import random
from math import floor

from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from knapsack import Knapsack
import numpy as np
from kinship_structure_navigation import Warlpiri_Subsection
from GA_helpers import init_hybrid_population, repair_individual
Population_Index_Dict = {
    0: "P1A",
    1: "P1B",
    2: "P2A",
    3: "P2B",
    4: "P3A",
    5: "P3B",
    6: "P4A",
    7: "P4B",
}
Population_Patromiety_Dict = {v: k for k, v in Population_Index_Dict.items()} # Reverse lookup table
# Fitness Function applied on all sets
# Suitable sets reproduce

# Fitness Function on each individual.
# Fit individuals are selected.
# Crossover (A pair of chromosomes) -> Mutation (each chromosome has a probibility)
# Calculate new Fitness for each individual



# Create an instance of a Knapsack problem
p_file = "Knapsack_Problems/problemInstances/n_600_c_1000000_g_10_f_0.1_eps_0.0001_s_100/test.in"
opt_file = "Knapsack_Problems/optima.csv"
knapsack = Knapsack(p_file, opt_file)

# Defining Problem Constraints
ITEMS_LENGTH = len(knapsack)
POPULATION_SIZE = 2000
P_CROSSOVER = 0.9
P_MUTATION = 0.005
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 10
ELITE_SIZE = 1 # 1 Elite per group. Total 8 Elites.
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
# Define 8 Populations with their index correlating to the Population_Index_Dict


populations = []
BASE_POP_SIZE = floor(POPULATION_SIZE / 8)

for _ in range(8):
    # Initialize each sub-population with the hybrid strategy
    pop = init_hybrid_population(toolbox, BASE_POP_SIZE, knapsack, feasible_ratio=0.5)
    populations.append(pop)

# Define a function to evaluate the total value of the selected items
def evaluate(individual):
    return knapsack.getTotalValue(individual), # return a tuple

# Register the countOnes function
toolbox.register("evaluate", evaluate)

# Create the genetic operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform, indpb=P_CROSSOVER)
toolbox.register("mutate", tools.mutFlipBit, indpb=P_MUTATION)

# Create the statistics object
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Register the statistics object
stats.register("max", np.max)
stats.register("avg", np.mean)

# Create the hall of fame object
#hof = tools.HallOfFame(HALL_OF_FAME_SIZE) # Not actually used at the moment

# ============================================
# This was the previous simple evolutionary algorithm simulation start. We must manually simulate for more finegrain control
# population, logbook = algorithms.eaSimple(toolbox.populationCreator(n=POPULATION_SIZE), toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
# ============================================

# Define the algorithm (NEW)

# Evaluate the original populations
for pop in populations:
    for individual in pop:
        individual.fitness.values = toolbox.evaluate(individual)


# A temporary dict to hold offspring before disposing of parents
offspring_dict = {i: [] for i in range(len(populations))}

# For Statistic Tracking:
max_values_history = []
avg_values_history = []
pop_max_history = {i: [] for i in range(8)}  # one list per population


# Begin Evolutionary Loop
for gen in range(MAX_GENERATIONS):
    offspring_dict = {i: [] for i in range(len(populations))} # Reset dict for new generation
    # Block of code to calculate who the other parent should be, and where the children should go
    for father_population_index in range(len(populations)):
        father_warlpiri_code = Population_Index_Dict[father_population_index]
        father_group_warlpiri = Warlpiri_Subsection(father_warlpiri_code) # Get Father's Warlpiri code. E.g. P1A
        ideal_partner_warlpiri = father_group_warlpiri.get_ideal_wife()
        ideal_partner_index = Population_Patromiety_Dict[ideal_partner_warlpiri.SemiPatrimoiety] # Convert Warlpiri -> Population Index
        child_warlpiri = father_group_warlpiri.get_child_node()
        child_assigned_population = Population_Patromiety_Dict[child_warlpiri.SemiPatrimoiety] # E.g. P1A -> Index[0]

        # Select the chosen populations
        father_group = toolbox.select(populations[father_population_index], len(populations[father_population_index]))  # Just as a test we will get the first population
        mother_group = toolbox.select(populations[ideal_partner_index], len(populations[ideal_partner_index])) # Get the accompanied mother group

        for p0, p1 in zip(father_group, mother_group): # Pair a father to a mother
            # Clone the parents so we can apply mutations without affecting original pair
            c1 = creator.Individual(p0)
            c2 = creator.Individual(p1)

            # Apply crossover (mating)
            if random.random() < P_CROSSOVER: # If probability hits
                toolbox.mate(c1, c2)

            # Apply Mutation
            toolbox.mutate(c1)
            toolbox.mutate(c2)

            # After crossover and mutation, repair any overweight individuals
            c1 = repair_individual(c1, knapsack)
            c2 = repair_individual(c2, knapsack)
            # Evaluate children fitness
            c1.fitness.values = toolbox.evaluate(c1)
            c2.fitness.values = toolbox.evaluate(c2)
            # c2.fitness.values = ()

            # Add new offspring to list of new children
            # if c1.fitness.values[0] > c2.fitness.values[0]:
            #     offspring_dict[child_assigned_population].extend([c1])
            # else:
            #     offspring_dict[child_assigned_population].extend([c2])
            offspring_dict[child_assigned_population].extend([c1]) # Add both children and crop infavourable later
            offspring_dict[child_assigned_population].extend([c2])

    # Before Elitism, track statistics
    all_children = [child for plist in offspring_dict.values() for child in plist]

    gen_max = max(ind.fitness.values[0] for ind in all_children)
    gen_avg = np.mean([ind.fitness.values[0] for ind in all_children])

    max_values_history.append(gen_max)
    avg_values_history.append(gen_avg)

    # Per-population tracking
    for pop_index, child_list in offspring_dict.items():
        if child_list:  # avoid empty list
            pop_best = max(ind.fitness.values[0] for ind in child_list)
        else:
            pop_best = float('-inf')
        pop_max_history[pop_index].append(pop_best)
    # ELITISM (SAVE 5 best parents from each generation)
    for target_population, child_list in offspring_dict.items():
        sorted_old = sorted(populations[target_population], key=lambda ind: ind.fitness.values[0], reverse=True)
        sorted_child = sorted(child_list, key=lambda ind: ind.fitness.values[0], reverse=True)
        elite = sorted_old[:ELITE_SIZE]
        populations[target_population][:] = elite + sorted_child[:len(populations[target_population]) - ELITE_SIZE]

    # Print best of children
    print(f"Gen {gen}: ", end="")
    for pop_index, pop in enumerate(populations):
        best = max(ind.fitness.values[0] for ind in pop)
        print(f"{Population_Index_Dict[pop_index]}={best}  ", end="")
    print()

# Print the best ever individual
# print("Best Individual: ", hof.items[0])
# print("Fitness:", knapsack.getTotalValue(hof.items[0]))
# knapsack.printItems(hof.items[0])

# Return the max and mean values
#maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

best_individual = None
best_fitness = float('-inf')

for pop in populations:
    for ind in pop:
        if ind.fitness.values[0] > best_fitness:
            best_fitness = ind.fitness.values[0]
            best_individual = ind

print("Best individual found:")
print(best_individual)
print("Fitness:", best_fitness)
knapsack.printItems(best_individual)

# Plot the max and mean vals
plt.plot(max_values_history, label="Max fitness")
plt.plot(avg_values_history, label="Avg fitness")
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness of Offspring Over Generations")
plt.show()
