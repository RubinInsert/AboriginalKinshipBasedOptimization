
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
from GA_helpers import init_hybrid_population, repair_individual, get_hamming_distance, calculate_population_diversity
import time
import keyboard
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
Wirlpiri_Child = {
    "P1A": "P1B",
    "P1B": "P1A",
    "P2A": "P2B",
    "P2B": "P2A",
    "P3A": "P3B",
    "P3B": "P3A",
    "P4A": "P4B",
    "P4B": "P4A"
}
Wirlpiri_Wife = {
    "P1A": "P4B",
    "P1B": "P2A",
    "P2A": "P1B",
    "P2B": "P3A",
    "P3A": "P2B",
    "P3B": "P4A",
    "P4A": "P3B",
    "P4B": "P1A"
}
Population_Patromiety_Dict = {v: k for k, v in Population_Index_Dict.items()} # Reverse lookup table


def run_WCGGA(knapsack, pop_size, crossover_probability, mutation_probability, max_time, max_generations, stagnation_limit,
              initialization_feasability, random_seed):
    ITEMS_LENGTH = len(knapsack)
    POPULATION_SIZE = pop_size
    P_CROSSOVER = crossover_probability
    P_MUTATION = mutation_probability
    MAX_GENERATIONS = max_generations
    MAX_TIME_S = max_time
    # Stagnation breaker
    STAGNATION_LIMIT = stagnation_limit  # If no new record in 20 gens, NUKE.
    # Trackers for EACH of the 8 groups independently
    group_best_fitness = [0] * 8
    group_stagnation_counters = [0] * 8
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
    hof = tools.HallOfFame(maxsize=10)
    # ====================================================================
    # BEGIN EVOLUTIONARY LOOP (DETERMINISTIC CROWDING)
    # ====================================================================
    diversity_history = []  # <--- NEW TRACKER
    group_diversity_history = []  # <--- OPTIONAL: Track diversity PER GROUP
    print("Starting Deterministic Crowding Evolution...")

    all_inds_init = [ind for pop in populations for ind in pop]
    hof.update(all_inds_init)

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

                # PARENT SELECTION (Warlpiri Rules)
                # Father is the individual currently sitting in this slot
                father = populations[father_idx][i]

                # Identify Mother's Group using Warlpiri Kinship
                father_code = Population_Index_Dict[father_idx]
                ideal_wife = Wirlpiri_Wife[father_code]
                mother_idx = Population_Patromiety_Dict[ideal_wife]

                # Mother is chosen RANDOMLY from the correct Warlpiri group
                # This ensures genetic mixing while obeying the strict marriage rule
                mother = random.choice(populations[mother_idx])

                # REPRODUCTION
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
                child_node = Wirlpiri_Child[father_code]
                child_dest_idx = Population_Patromiety_Dict[child_node]

                # --- D. CROWDING REPLACEMENT (Restricted Tournament Selection) ---
                # 1. Define how many candidates to check (Crowding Factor)
                #    Larger window = Stronger crowding (more diversity preservation)
                #    Smaller window = Faster convergence
                CROWDING_WINDOW_SIZE = 3

                # 2. Pick random candidates from the destination group
                #    We include the current index 'i' to ensure steady turnover
                candidate_indices = random.sample(range(BASE_POP_SIZE), CROWDING_WINDOW_SIZE - 1)
                candidate_indices.append(i)

                # 3. Find the candidate most similar to the child (Lowest Hamming Distance)
                closest_incumbent_idx = -1
                min_distance = float('inf')

                target_pop = populations[child_dest_idx]

                for idx in candidate_indices:
                    dist = get_hamming_distance(child, target_pop[idx])
                    if dist < min_distance:
                        min_distance = dist
                        closest_incumbent_idx = idx

                # 4. Compete ONLY against the most similar individual found
                closest_incumbent = target_pop[closest_incumbent_idx]

                if child.fitness.values[0] >= closest_incumbent.fitness.values[0]:
                    # Update In-Place: Child replaces its closest genetic relative
                    new_ind = creator.Individual(child)
                    new_ind.fitness.values = child.fitness.values
                    target_pop[closest_incumbent_idx] = new_ind
        for i, pop in enumerate(populations):
            current_group_max = max(ind.fitness.values[0] for ind in pop)

            # Check if this specific group beat its own record
            if current_group_max > group_best_fitness[i]:
                group_best_fitness[i] = current_group_max
                group_stagnation_counters[i] = 0  # Reset
            else:
                group_stagnation_counters[i] += 1  # Increment

        # 1. UPDATE STATISTICS & HOF
        # Flatten all 8 groups into one list
        all_inds = [ind for pop in populations for ind in pop]

        # Update the Hall of Fame with the current generation
        hof.update(all_inds)  # <--- NEW: Automatically keeps the best, discards the rest

        # Check for Convergence / Stagnation
        least_stagnant_count = min(group_stagnation_counters)

        if least_stagnant_count >= STAGNATION_LIMIT:
            print(f"\n NUKE RELEASED \n")

            # A. Find the global best fitness individual
            best_ind_global = hof[0]
            best_fitness_global = -1


            # Create a deep copy of best individual
            best_ind = creator.Individual(best_ind_global)
            best_ind.fitness.values = best_ind_global.fitness.values

            # B. Nuke Everything (PURE RANDOM RESET)
            for i in range(len(populations)):
                populations[i] = init_hybrid_population(
                    toolbox,
                    BASE_POP_SIZE,
                    knapsack,
                    feasible_ratio=0.0  # <--- CRITICAL: Pure random reset
                )
                # Repair & Evaluate
                for ind in populations[i]:
                    ind = repair_individual(ind, knapsack)
                    ind.fitness.values = toolbox.evaluate(ind)

            # place
            populations[0][0] = best_ind

            # Groups 1-7: Mutant Noahs
            for i in range(1, 8):
                mutant_best_ind = creator.Individual(best_ind)
                is_different = False
                while not is_different:
                    toolbox.mutate(mutant_best_ind)
                    if mutant_best_ind != best_ind:
                        is_different = True

                mutant_best_ind = repair_individual(mutant_best_ind, knapsack)
                mutant_best_ind.fitness.values = toolbox.evaluate(mutant_best_ind)
                populations[i][0] = mutant_best_ind

            for i in range(8): # Reset Stagnation detection vars
                current_new_max = max(ind.fitness.values[0] for ind in populations[i])
                group_best_fitness[i] = current_new_max
                group_stagnation_counters[i] = 0

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
        # 1. Global Diversity (Whole Population)
        # Flatten the subpopulations into one list for calculation
        all_inds = [ind for pop in populations for ind in pop]
        global_div = calculate_population_diversity(all_inds)
        diversity_history.append(global_div)

        # 2. (Optional) Group Specific Diversity - To prove "Different Camps"
        # Only calculate this every 10 gens to save time
        if gen % 10 == 0:
            current_group_divs = [calculate_population_diversity(pop) for pop in populations]
            group_diversity_history.append(current_group_divs)
        # Early Break - For testing
        if keyboard.is_pressed("q"):
            print("Early Stopping")
            break
# ====================================================================
# FINAL SELECTION & OPTIMIZATION
# ====================================================================

    return {
        "best_individual": hof[0],
        "max_values": max_values_history,
        "avg_values": avg_values_history,
        "diversity_history": diversity_history,
        "group_diversity_history": group_diversity_history
    }
# Fitness Function applied on all sets
# Suitable sets reproduce

# Fitness Function on each individual.
# Fit individuals are selected.
# Crossover (A pair of chromosomes) -> Mutation (each chromosome has a probibility)
# Calculate new Fitness for each individual





# Defining Problem Constraints


