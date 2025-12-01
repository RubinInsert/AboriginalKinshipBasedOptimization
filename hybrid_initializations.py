import random
import numpy as np


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
        if random.random() < 0.1:  # 10% chance to skip
            continue

        if current_weight + weight <= knapsack_instance.maxCapacity:
            individual[index] = 1
            current_weight += weight

        # Optimization: Stop once the knapsack is mostly full
        if current_weight > knapsack_instance.maxCapacity * 0.95:
            break

    return individual