import os
import csv


class Knapsack:
    def __init__(self, file_path, optima_file_path=None):
        """
        file_path: Path to the instance file.
        optima_file_path: Path to the optimal value file (CSV or single value).
        """
        self.file_path = file_path
        self.maxCapacity = 0
        self.items = []  # List of (weight, value)
        self.optimal_value = None

        # 1. Determine Problem ID (Smart naming)
        filename = os.path.basename(file_path)
        if filename.lower() in ["test.in", "test.txt", "problem.in"]:
            # Use Folder Name for Standard/Jooken files
            self.problem_name = os.path.basename(os.path.dirname(file_path))
        else:
            # Use Filename for Pisinger files
            self.problem_name = filename

        # 2. Load the problem instance
        self._load_instance(file_path)

        # 3. Load the optimal value if provided
        if optima_file_path:
            self._load_optimal_value(optima_file_path)

    def __len__(self):
        return len(self.items)

    def _load_instance(self, file_path):
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"File {file_path} is empty.")

        # Check Header: 1 integer (Standard) or 2 integers (Pisinger)
        header_parts = lines[0].split()

        if len(header_parts) == 2:
            self._load_pisinger_format(lines)
        elif len(header_parts) == 1:
            self._load_standard_format(lines)
        else:
            raise ValueError("Unknown file header format.")

    def _load_pisinger_format(self, lines):
        # Header: n c
        header = lines[0].split()
        num_items = int(header[0])
        self.maxCapacity = float(header[1])

        self.items = []
        # Read exactly num_items lines to skip the trailing solution vector
        for i in range(1, num_items + 1):
            parts = lines[i].split()
            # Format: Value Weight
            if len(parts) >= 2:
                value = float(parts[0])
                weight = float(parts[1])
                self.items.append((weight, value))

    def _load_standard_format(self, lines):
        # Header: n
        num_items = int(lines[0])
        self.items = []
        for i in range(1, num_items + 1):
            parts = lines[i].split()
            # Format: [ID, Profit, Weight] or [Profit, Weight]
            if len(parts) >= 3:
                value = float(parts[1])
                weight = float(parts[2])
            else:
                value = float(parts[0])
                weight = float(parts[1])
            self.items.append((weight, value))

        self.maxCapacity = float(lines[-1])

    def _load_optimal_value(self, optima_path):
        self.optimal_value = None

        # STRATEGY A: Single Value File (Pisinger Optimal Files)
        try:
            with open(optima_path, 'r') as f:
                content = f.read().strip()
                # Check if file contains ONLY a number (no commas/newlines)
                if content.replace('.', '', 1).isdigit() and ',' not in content:
                    self.optimal_value = float(content)
                    print(f"Loaded Optimal (Direct): {self.optimal_value}")
                    return
        except Exception:
            pass

            # STRATEGY B: CSV Lookup (Standard Optimal Files)
        try:
            with open(optima_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Check if problem name matches the CSV key
                    if len(row) >= 2 and self.problem_name in row[0]:
                        self.optimal_value = float(row[1])
                        print(f"Loaded Optimal (CSV): {self.optimal_value}")
                        return
        except Exception as e:
            print(f"CSV Lookup Warning: {e}")

        print(f"Warning: Optimal value not found for '{self.problem_name}'")
    def getTotalWeight(self, individual):
        """
        Calculates the total weight of the selected items in the individual.
        Assumes self.items is a list of (weight, value) tuples.
        """
        total_weight = 0
        for i, bit in enumerate(individual):
            if bit:
                # Add the weight (index 0 of the item tuple)
                total_weight += self.items[i][0]
        return total_weight
    def getTotalValue(self, binaryChoiceList):
        totalWeight = 0
        totalValue = 0
        for i, choice in enumerate(binaryChoiceList):
            if choice:
                weight, value = self.items[i]
                totalWeight += weight
                totalValue += value

        if totalWeight <= self.maxCapacity:
            return totalValue
        else:
            # Penalty
            penalty = (totalWeight - self.maxCapacity) * 20
            return totalValue - penalty


# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- 1. Testing Standard Format ---")
    # Point this to your standard test.in
    p_std = "Knapsack_Problems/problemInstances/n_600_c_1000000_g_10_f_0.1_eps_0.0001_s_100/test.in"
    opt_std = "Knapsack_Problems/optima.csv"

    if os.path.exists(p_std):
        k1 = Knapsack(p_std, opt_std)
        print(f"Success: {k1.problem_name} | Opt: {k1.optimal_value}")
    else:
        print("Standard file not found.")

    print("\n--- 2. Testing Pisinger Format ---")
    # Point this to your Pisinger file
    p_pis = r"Pisinger_Knapsack_Problems\large_scale\knapPI_1_100_1000_1"
    opt_pis = r"Pisinger_Knapsack_Problems\large_scale-optimum\knapPI_1_100_1000_1"

    if os.path.exists(p_pis):
        k2 = Knapsack(p_pis, opt_pis)
        # Verify first item (Should be 485 weight, 94 value)
        print(f"First Item: {k2.items[0]}")
        print(f"Success: {k2.problem_name} | Opt: {k2.optimal_value}")
    else:
        print("Pisinger file not found.")