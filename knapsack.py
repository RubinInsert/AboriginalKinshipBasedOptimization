import os
import csv


class Knapsack:
    def __init__(self, file_path, optima_file_path=None):
        """
        file_path: Path to the .in or .txt instance file
        optima_file_path: Path to the optima.csv file (optional, for validation)
        """
        dir_path = os.path.dirname(file_path)
        self.maxCapacity = 0
        self.items = []  # List of (weight, value)
        self.optimal_value = None
        self.problem_name = os.path.basename(dir_path)

        self._load_from_file(file_path)

        if optima_file_path:
            self._load_optimal_value(optima_file_path)

    def __len__(self):
        return len(self.items)

    def _load_from_file(self, file_path):
        """Parse the Jooken/Pisinger style knapsack file."""
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Line 1: Number of items
        num_items = int(lines[0])

        # Lines 2 to N+1: Item details (ID, Profit, Weight)
        # We skip the ID (parts[0]) as requested
        self.items = []
        for i in range(1, num_items + 1):
            parts = lines[i].split()
            # Standard format: [ID, Profit, Weight]
            # Some variations might be [Profit, Weight], so we check length
            if len(parts) >= 3:
                value = float(parts[1])
                weight = float(parts[2])
            else:
                # Fallback for simpler formats
                value = float(parts[0])
                weight = float(parts[1])

            self.items.append((weight, value))

        # Last Line: Capacity
        self.maxCapacity = float(lines[-1])

    def _load_optimal_value(self, optima_csv_path):
        """Looks up the optimal value from a separate CSV file."""
        try:
            with open(optima_csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Row format: instance_name, optimal_value
                    print (self.problem_name)
                    if self.problem_name in row[0]:
                        self.optimal_value = float(row[1])
                        break
        except Exception as e:
            print(f"Warning: Could not load optimal value. {e}")

        print(
            f"Loaded {self.problem_name}: {len(self.items)} items, capacity = {self.maxCapacity}, known optimal = {self.optimal_value}")

    def getTotalValue(self, binaryChoiceList):
        """
        Calculates fitness.
        Returns actual value if valid.
        Returns penalized value if weight > capacity.
        """
        totalWeight = 0
        totalValue = 0

        # Determine strict or soft penalty
        # For Genetic Algorithms, a soft penalty is often better to allow
        # the population to traverse the boundary of feasibility.

        for i, choice in enumerate(binaryChoiceList):
            if choice:  # Assuming binary 1/0 or True/False
                weight, value = self.items[i]
                totalWeight += weight
                totalValue += value

        if totalWeight <= self.maxCapacity:
            return totalValue
        else:
            # Standard Linear Penalty
            penalty = (totalWeight - self.maxCapacity) * 1000
            return totalValue - penalty
    def printItems(self, binaryChoiceList):
        totalWeight = 0
        totalValue = 0
        print(f"--- Solution Details for {self.problem_name} ---")
        for i, choice in enumerate(binaryChoiceList):
            if choice > 0:
                weight, value = self.items[i]
                totalWeight += weight
                totalValue += value
                print(f"- Item {i}: Weight={weight}, Value={value}")

        print(f"--- Summary ---")
        print(f"Total Weight: {totalWeight} / {self.maxCapacity}")
        print(f"Total Value:  {totalValue}")
        if self.optimal_value:
            print(f"Optimality Gap: {self.optimal_value - totalValue}")


if __name__ == "__main__":
    # Example Usage
    # Ensure you have downloaded the dataset mentioned above

    # Path to a specific problem file
    p_file = "Knapsack_Problems/problemInstances/n_600_c_1000000_g_10_f_0.1_eps_0.0001_s_100/test.in"

    # Path to the solution key
    opt_file = "Knapsack_Problems/optima.csv"

    try:
        knapsack = Knapsack(p_file, opt_file)

        # Test with a dummy solution (all items selected)
        dummy_sol = [1] * len(knapsack)
        print(f"Fitness of taking all items: {knapsack.getTotalValue(dummy_sol)}")

    except FileNotFoundError:
        print("Please set the correct file paths to run the test.")
