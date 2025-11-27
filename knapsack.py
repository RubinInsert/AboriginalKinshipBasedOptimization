import numpy as np

class Knapsack:
    def __init__(self):
        self.maxCapacity = 0

        self.items = [] # List of items available to the knapsack

        self.__initData()

    def __len__(self): # Return the total number of items in the knapsack
        return len(self.items)

    def __initData(self):
        self.items = [ # Name, Value, Weight
            ("Gold Coin", 60, 10),
            ("Silver Coin", 40, 8),
            ("Bronze Coin", 25, 7),
            ("Emerald Gem", 90, 13),
            ("Ruby Gem", 95, 14),
            ("Sapphire Gem", 85, 11),
            ("Diamond Shard", 110, 15),
            ("Magic Scroll", 70, 5),
            ("Leather Boots", 30, 12),
            ("Iron Sword", 50, 20),
            ("Steel Shield", 65, 25),
            ("Health Potion", 40, 3),
            ("Mana Potion", 45, 3),
            ("Elixir", 90, 4),
            ("Magic Ring", 100, 2),
            ("Magic Amulet", 120, 3),
            ("Torch", 20, 6),
            ("Rope", 15, 9),
            ("Grappling Hook", 55, 10),
            ("Lantern", 35, 8),
            ("Compass", 50, 4),
            ("Map", 30, 2),
            ("Crossbow", 80, 18),
            ("Bolts Pack", 22, 5),
            ("Herbs Bundle", 28, 4),
            ("Crystal Orb", 130, 12),
            ("Enchanted Cloak", 140, 9),
            ("Book of Spells", 95, 10),
            ("Traveler's Hat", 18, 3),
            ("Blacksmith Tools", 60, 22),
            ("Carpenter Tools", 55, 20),
            ("Herbal Kit", 48, 6),
            ("Fire Bomb", 75, 5),
            ("Ice Bomb", 78, 5),
            ("Smoke Bomb", 32, 3),
            ("Lockpick Set", 42, 2),
            ("Spyglass", 58, 7),
            ("Telescope", 80, 12),
            ("Gemstone Box", 150, 16),
            ("Ancient Relic", 200, 18)
        ]
        self.maxCapacity = 400
    def getTotalValue(self, binaryChoiceList): # Calculate the value of a set of choices. If the choice is overweight, the set's value is penalised significantly
        totalWeight = totalValue = 0
        for i in range(len(binaryChoiceList)):
            item, weight, value = self.items[i]
            totalWeight += binaryChoiceList[i] * weight # if choice isnt chosen, multiply by zero essentially skipping item
            totalValue += binaryChoiceList[i] * value
        if totalWeight <= self.maxCapacity:
            return totalValue
        else:
            penalty = (totalWeight - self.maxCapacity) * 10
        return totalValue - penalty
    def printItems(self, binaryChoiceList):
        totalWeight = totalValue = 0
        for i in range(len(binaryChoiceList)):
            item, weight, value = self.items[i]
            if binaryChoiceList[i] > 0:
                totalWeight += weight
                totalValue += value
                print("- Adding {}: weight = {}, value = {}, accumulated weight = {}, accumulated value = {}".format(item, weight, value, totalWeight, totalValue))
        print("- Total weight = {}, Total value = {}".format(totalWeight, totalValue))

