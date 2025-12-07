import random
import numpy as np
from deap import creator


def create_feasible_individual(knapsack_instance):
    """
    Creates a guaranteed feasible solution using the Greedy by Density heuristic.
    """
    # 1. Prepare items with density (Value/Weight)
    # Item format in Knapsack class is (weight, value)
    indexed_items = []
    for i, (weight, value) in enumerate(knapsack_instance.items):
        if weight > 0:
            density = value / weight
            indexed_items.append((density, weight, value, i))
        else:
            # Handle zero weight items by assigning infinite density
            indexed_items.append((float('inf'), weight, value, i))

    # 2. Sort by density (descending)
    indexed_items.sort(key=lambda x: x[0], reverse=True)

    # 3. Greedily select items up to capacity (with some randomization)
    individual = [0] * len(knapsack_instance.items)
    current_weight = 0

    for density, weight, value, index in indexed_items:
        # Introduce a small chance of skipping the item for added diversity
        # RANKED CHOICE
        if random.random() < 0.1:  # 10% chance to skip
            continue

        if current_weight + weight <= knapsack_instance.maxCapacity:
            individual[index] = 1
            current_weight += weight

        # Optimization: Stop once the knapsack is mostly full
        if current_weight > knapsack_instance.maxCapacity * 0.95:
            break

    return individual
def init_hybrid_population(toolbox, pop_size, knapsack_instance, feasible_ratio=0.5):
    """Initializes a population with a mix of random and heuristically feasible individuals."""

    n_feasible = int(pop_size * feasible_ratio)
    population = []

    # Generate Feasible Individuals
    for _ in range(n_feasible):
        ind_list = create_feasible_individual(knapsack_instance)
        population.append(creator.Individual(ind_list))

    # Generate Random Individuals
    random_pop = toolbox.populationCreator(n=pop_size - n_feasible)
    population.extend(random_pop)

    random.shuffle(population)
    return population

def repair_individual(ind, knapsack):
    """
    Ensures an individual is feasible (total weight <= capacity).
    Removes items with the lowest value/weight ratio first.
    """
    # Compute current total weight
    total_weight = 0
    for i, bit in enumerate(ind):
        if bit:
            w, v = knapsack.items[i]
            total_weight += w

    # Already feasible? nothing to do.
    # if total_weight <= knapsack.maxCapacity: # We want to ensure the greedy fill still occurs on feasible individuals
    #     return ind
    is_originally_overweight = total_weight > knapsack.maxCapacity
    # Build list of (index, value/weight ratio)
    items_present = []
    for i, bit in enumerate(ind):
        if bit:
            w, v = knapsack.items[i]
            if w > 0:
                items_present.append((i, v / w))
            else:
                items_present.append((i, float('inf')))

    # Sort items by worst ratio first (we want to remove worst items)
    items_present.sort(key=lambda x: x[1])  # ascending ratio
    if total_weight <= knapsack.maxCapacity:
    # Remove items until feasible
        for idx, ratio in items_present:
            ind[idx] = 0  # remove item
            w, v = knapsack.items[idx]
            total_weight -= w
            if total_weight <= knapsack.maxCapacity:
                break
    # # FILL PHASE
    # if is_originally_overweight:
    #     items_available_to_add = []
    #
    #     for i, bit in enumerate(ind):
    #         if not bit:  # Only consider items currently NOT selected (i.e., item index 'i' has ind[i]=0)
    #             w, v = knapsack.items[i]
    #
    #             # Calculate ratio, handling weight=0 case
    #             if w > 0:
    #                 ratio = v / w
    #             else:
    #                 ratio = float('inf')
    #
    #             # Store (index, ratio, weight) for adding
    #             items_available_to_add.append((i, ratio, w))
    #
    #     # Sort by best ratio first (descending ratio)
    #     items_available_to_add.sort(key=lambda x: x[1], reverse=True)
    #
    #     # Greedily add items
    #     for idx, ratio, w in items_available_to_add:
    #         # Check if the item fits in the remaining capacity
    #         if total_weight + w <= knapsack.maxCapacity and random.random() < 0.5:
    #             ind[idx] = 1  # Add item
    #             total_weight += w

    return ind