import random
import numpy as np
from deap import creator
from operator import xor

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
        if random.random() < 0.3:  # 10% chance to skip
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




def get_hamming_distance(ind1, ind2):
    # Using map with operator.xor is often faster than a python for-loop/zip
    # for large lists in standard CPython
    return sum(map(xor, ind1, ind2))
def repair_individual_oldest(ind, knapsack):
    """
    Ensures an individual is feasible (total weight <= capacity).
    Removes items with the lowest value/weight ratio first.
    """
    # Compute current total weight
    total_weight = sum(knapsack.items[i][0] for i, bit in enumerate(ind) if bit)
    # Already feasible? nothing to do.
    # if total_weight <= knapsack.maxCapacity: # We want to ensure the greedy fill still occurs on feasible individuals
    #     return ind
    is_originally_overweight = total_weight > knapsack.maxCapacity
    # Build list of (index, value/weight ratio)
    # Get indices of items currently in the bag
    items_present = [i for i, gene in enumerate(ind) if gene == 1]
    random.shuffle(items_present)
    # Sort items by worst ratio first (we want to remove worst items)
    #items_present.sort(key=lambda x: x[1])  # ascending ratio
    # if total_weight <= knapsack.maxCapacity:
    # # Remove items until feasible - GREEDY
    # #     for idx, ratio in items_present:
    # #         ind[idx] = 0  # remove item
    # #         w, v = knapsack.items[idx]
    # #         total_weight -= w
    # #         if total_weight <= knapsack.maxCapacity:
    # #             break

    # Remove items until feasible randomized
    idx = 0
    while total_weight > knapsack.maxCapacity and idx < len(items_present):
        remove_index = items_present[idx]
        w, v = knapsack.items[remove_index]

        ind[remove_index] = 0
        total_weight -= w
        idx += 1  # Just move to the next item in the shuffled list
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


def repair_individual_old(ind, knapsack):
    """
    Hybrid Repair:
    1. Stochastic Removal: Removes items to fit capacity. Biased towards removing
       inefficient items, but not purely greedy (avoids traps).
    2. Greedy Fill: Fills remaining space with best available items.
    """
    total_weight = knapsack.getTotalWeight(ind)

    # --- PHASE 1: REMOVAL (If overweight) ---
    if total_weight > knapsack.maxCapacity:
        # Get indices of items currently in the bag
        items_in_bag = [i for i, bit in enumerate(ind) if bit == 1]

        # We need to remove items until we fit.
        # Instead of sorting (slow/greedy), we use a "Tournament of 2" for removal.
        # We pick 2 items, and remove the one with the WORST v/w ratio.
        # This is fast, stochastic, but smart.

        while total_weight > knapsack.maxCapacity:
            # Pick 2 random items from the bag
            if len(items_in_bag) >= 2:
                candidates = random.sample(items_in_bag, 2)
                item_a = candidates[0]
                item_b = candidates[1]

                # Compare Value/Weight ratios
                w_a, v_a = knapsack.items[item_a]
                w_b, v_b = knapsack.items[item_b]
                r_a = v_a / w_a if w_a > 0 else 0
                r_b = v_b / w_b if w_b > 0 else 0

                # Identify the "loser" (worse ratio)
                victim = item_a if r_a < r_b else item_b
            else:
                # Fallback if only 1 item left (rare)
                victim = items_in_bag[0]

            # Remove the victim
            ind[victim] = 0
            w_victim, _ = knapsack.items[victim]
            total_weight -= w_victim
            items_in_bag.remove(victim)

    # --- PHASE 2: GREEDY FILL (Always try to pack more!) ---
    # Now that we are valid, let's see if we can squeeze in the best remaining items.

            # 1. Identify items currently NOT in the bag
            items_out_indices = [i for i, bit in enumerate(ind) if bit == 0]

            # 2. Prepare candidates: (index, weight, ratio)
            items_to_add = []
            for i in items_out_indices:
                w, v = knapsack.items[i]
                if w > 0:
                    ratio = v / w
                    items_to_add.append((i, w, ratio))

            # 3. Sort by Ratio Descending (Strict Greedy)
            items_to_add.sort(key=lambda x: x[2], reverse=True)

            # 4. Windowed Fill (The Magic Fix)
            # Instead of iterating strict 0..N, we iterate in small random chunks.
            # This preserves "Goodness" but breaks "Determinism".
            WINDOW_SIZE = 5

            i = 0
            while i < len(items_to_add):
                # Grab a chunk of the best remaining items
                chunk = items_to_add[i: i + WINDOW_SIZE]

                # Shuffle this chunk (Randomness!)
                random.shuffle(chunk)

                # Try to add items from this shuffled chunk
                for index, w, ratio in chunk:
                    if total_weight + w <= knapsack.maxCapacity:
                        ind[index] = 1
                        total_weight += w

                        # Optimization: If almost full, stop early
                        if knapsack.maxCapacity - total_weight < 1:
                            return ind

                i += WINDOW_SIZE

            return ind

    return ind


def repair_individual(ind, knapsack):
    """
    Clumsy Hybrid Repair:
    - Removal: 10% chance to 'drop the wrong item' (Exploration).
    - Fill: Larger Window (20) to find non-greedy packing combinations.
    """
    total_weight = knapsack.getTotalWeight(ind)

    # --- PHASE 1: CLUMSY REMOVAL ---
    if total_weight > knapsack.maxCapacity:
        items_in_bag = [i for i, bit in enumerate(ind) if bit == 1]

        while total_weight > knapsack.maxCapacity:
            # Tournament of 2
            if len(items_in_bag) >= 2:
                candidates = random.sample(items_in_bag, 2)
                item_a, item_b = candidates[0], candidates[1]

                w_a, v_a = knapsack.items[item_a]
                w_b, v_b = knapsack.items[item_b]
                r_a = v_a / w_a if w_a > 0 else 0
                r_b = v_b / w_b if w_b > 0 else 0

                # LOGIC CHANGE: 10% chance to act irrationally
                # This prevents the "Magnet Effect" of always keeping the best items.
                if random.random() < 0.10:
                    victim = item_a if r_a > r_b else item_b  # Remove the BETTER one (Oops!)
                else:
                    victim = item_a if r_a < r_b else item_b  # Remove the WORSE one (Standard)
            else:
                victim = items_in_bag[0]

            ind[victim] = 0
            w_victim = knapsack.items[victim][0]
            total_weight -= w_victim
            items_in_bag.remove(victim)

    # --- RE-SYNC WEIGHT (Float Drift Protection) ---
    total_weight = knapsack.getTotalWeight(ind)

    # --- PHASE 2: WIDER WINDOW FILL ---
    items_out_indices = [i for i, bit in enumerate(ind) if bit == 0]

    items_to_add = []
    for i in items_out_indices:
        w, v = knapsack.items[i]
        if total_weight + w <= knapsack.maxCapacity:
            if w > 0:
                ratio = v / w
                items_to_add.append((i, w, ratio))

    # Sort by Ratio Descending
    items_to_add.sort(key=lambda x: x[2], reverse=True)

    # LOGIC CHANGE: Increase Window Size significantly
    # A size of 20 allows "mediocre" items to be tested in combination
    WINDOW_SIZE = 20

    i = 0
    while i < len(items_to_add):
        chunk = items_to_add[i: i + WINDOW_SIZE]
        random.shuffle(chunk)  # Shuffle the top 20 candidates

        for index, w, ratio in chunk:
            if total_weight + w <= knapsack.maxCapacity:
                ind[index] = 1
                total_weight += w
                if knapsack.maxCapacity - total_weight < 0.001:
                    return ind
        i += WINDOW_SIZE

    return ind


import numpy as np


def calculate_population_diversity(population):
    """
    Calculates diversity efficiently using column-wise variance.
    Returns a float: 0.0 (Identical clones) to 0.25 (Maximum chaos).
    """
    if not population: return 0.0

    # Convert list of individuals to boolean numpy array for speed
    # (Assuming individuals are lists of 0s and 1s)
    pop_matrix = np.array([ind for ind in population], dtype=np.uint8)

    # Calculate the fraction of 1s for each item (column)
    p = np.mean(pop_matrix, axis=0)

    # Variance for binary data = p * (1 - p)
    # Sum of variance across all items gives total population diversity
    diversity_score = np.sum(p * (1 - p))

    # Normalize by number of items so it's comparable across different problem sizes
    return diversity_score / pop_matrix.shape[1]