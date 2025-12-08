
# Changed Tourn Size from 3 to 2, so elites are less dominant
# Changed mutation quantity from a constant value to 2/ITEMS_LENGTH
# Changed Mating from cxUniform to cxTwoPoint to more accurately represent mating, and provide better results
# Current Best on n_1200_c_1000000_g_10_f_0.3_eps_0.0001_s_300: 1036019
# Consider adding Greedy Fill in repair operator



# Added Greedy Fill in repair operator
# Current Best on n_1200_c_1000000_g_10_f_0.3_eps_0.0001_s_300: 1036114.0 (OPTIMAL REACHED)
# KNAPSACK PROBLEMS: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html


# KEY FEATURES:
# 8 Groups
# Uses cxTwoPoint Crossover
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
# + Start from the initialization of the population.
# + Measure the average quality of initial population -> The improvement over time.
# Possibly update the generic evolutionary algorithm to include the above key features for an additional benchmark
# Generic Co-evolutionary algorithm
# Questions:
# - What Co-evolutionary algorithm would be best to benchmark against? E.g. competitive or cooperative? I presume competitive?
#  + Competitive Co-Evolutionary Algorithm
# + The children and parents remain in the same sub-pop.
# + Check papers for paremeters, Clockwise or Random.
# - How should the generic co-evolutionary algorithm function? should the children just be placed in random groups? are parents chosen from their same group?

# + Display optimality as a percentage rather than subtraction: Allows comparison across instances
# + As algorithm converges P_CROSSOVER--, P_MUTATION++
import random
from math import floor

from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from knapsack import Knapsack
import numpy as np
from kinship_structure_navigation import Warlpiri_Subsection
from GA_helpers import init_hybrid_population, repair_individual
import time
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


def run_WCGGA(knapsack, pop_size, crossover_probability, mutation_probability, max_time, max_generations, elite_size,
              initialization_feasability, random_seed):
    ITEMS_LENGTH = len(knapsack)
    POPULATION_SIZE = pop_size
    P_CROSSOVER = crossover_probability
    P_MUTATION = mutation_probability
    MAX_GENERATIONS = max_generations
    MAX_TIME_S = max_time
    # Note: ELITE_SIZE is not used in Crowding because "Elite" status is maintained automatically by the competition.

    # Random Seed
    RANDOM_SEED = random_seed
    random.seed(RANDOM_SEED)

    # 1. Setup DEAP Toolbox
    toolbox = base.Toolbox()
    toolbox.register('zeroOrOne', random.randint, 0, 1)

    # Check if classes already exist to avoid DEAP warnings on re-runs
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ITEMS_LENGTH)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # 2. Initialization
    start_time = time.time()
    populations = []
    BASE_POP_SIZE = floor(POPULATION_SIZE / 8)

    for _ in range(8):
        pop = init_hybrid_population(toolbox, BASE_POP_SIZE, knapsack, feasible_ratio=initialization_feasability)
        populations.append(pop)

    def evaluate(individual):
        return knapsack.getTotalValue(individual),

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=P_MUTATION)

    # Evaluate initial population
    for pop in populations:
        for individual in pop:
            individual.fitness.values = toolbox.evaluate(individual)

    # Statistics Tracking
    max_values_history = []
    avg_values_history = []

    # ====================================================================
    # BEGIN EVOLUTIONARY LOOP (DETERMINISTIC CROWDING)
    # ====================================================================
    print("Starting Deterministic Crowding Evolution...")

    for gen in range(MAX_GENERATIONS):
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_TIME_S:
            print(f"\nTime Limit Reached.")
            break

        # Iterate through every 'slot' index in the sub-populations
        # (This ensures every individual gets a chance to reproduce)
        for i in range(BASE_POP_SIZE):

            # Iterate through every Sub-Group (Father's Group)
            for father_idx in range(len(populations)):

                # --- A. PARENT SELECTION (Warlpiri Rules) ---
                # Father is the individual currently sitting in this slot
                father = populations[father_idx][i]

                # Identify Mother's Group using Warlpiri Kinship
                father_code = Population_Index_Dict[father_idx]
                father_node = Warlpiri_Subsection(father_code)
                ideal_wife = father_node.get_ideal_wife()
                mother_idx = Population_Patromiety_Dict[ideal_wife.SemiPatrimoiety]

                # Mother is chosen RANDOMLY from the correct Warlpiri group
                # This ensures genetic mixing while obeying the strict marriage rule
                mother = random.choice(populations[mother_idx])

                # --- B. REPRODUCTION ---
                child = creator.Individual(father)  # Inherit from father initially
                child_mother_genes = creator.Individual(mother)  # Temp copy for crossover

                # Crossover
                if random.random() < P_CROSSOVER:
                    toolbox.mate(child, child_mother_genes)

                # Mutation
                toolbox.mutate(child)

                # Repair & Evaluate
                child = repair_individual(child, knapsack)
                child.fitness.values = toolbox.evaluate(child)

                # --- C. CHILD PLACEMENT (Warlpiri Rules) ---
                # Determine which group the child belongs to (Child follows Father's Totem path)
                child_node = father_node.get_child_node()
                child_dest_idx = Population_Patromiety_Dict[child_node.SemiPatrimoiety]

                # --- D. CROWDING REPLACEMENT ---
                # The child fights the person CURRENTLY occupying slot 'i' in the destination group.
                # It does NOT fight the whole population.
                incumbent = populations[child_dest_idx][i]

                if child.fitness.values[0] > incumbent.fitness.values[0]:
                    # Update In-Place: Child replaces the incumbent
                    # We create a deep copy to be safe
                    new_ind = creator.Individual(child)
                    new_ind.fitness.values = child.fitness.values
                    populations[child_dest_idx][i] = new_ind

        # ====================================================================
        # STATISTICS & LOGGING
        # ====================================================================
        # Calculate stats based on the updated population
        all_inds = [ind for pop in populations for ind in pop]
        gen_max = max(ind.fitness.values[0] for ind in all_inds)
        gen_avg = np.mean([ind.fitness.values[0] for ind in all_inds])

        max_values_history.append(gen_max)
        avg_values_history.append(gen_avg)

        print(f"Gen {gen}: ", end="")
        for pop_index, pop in enumerate(populations):
            best = max(ind.fitness.values[0] for ind in pop)
            print(f"{Population_Index_Dict[pop_index]}={best:.0f}  ", end="")
        print()

    # Final Output
    best_individual = None
    best_fitness = float('-inf')
    for pop in populations:
        for ind in pop:
            if ind.fitness.values[0] > best_fitness:
                best_fitness = ind.fitness.values[0]
                best_individual = ind

    print("\nBest individual found:")
    print("Fitness:", best_fitness)

    plt.plot(max_values_history, label="Max fitness")
    plt.plot(avg_values_history, label="Avg fitness")
    plt.legend()
    plt.show()

    return {
        "best_individual": best_individual,
        "max_values": max_values_history,
        "avg_values": avg_values_history
    }

# Fitness Function applied on all sets
# Suitable sets reproduce

# Fitness Function on each individual.
# Fit individuals are selected.
# Crossover (A pair of chromosomes) -> Mutation (each chromosome has a probibility)
# Calculate new Fitness for each individual





# Defining Problem Constraints


