import numpy as np
SemiPatrimoiety_Dictionary = {
    "P1A": "M1A",
    "P1B": "M2A",
    "P2A": "M1B",
    "P2B": "M2B",
    "P3A": "M1C",
    "P3B": "M2C",
    "P4A": "M1D",
    "P4B": "M2D"
}
Matrimoiety_Dictionary = {v: k for k, v in SemiPatrimoiety_Dictionary.items()}
def next_matrimoiety(c):
    return chr((ord(c) - ord('A') + 1) % 4 + ord('A')) # Get the next character in the order: A -> B -> C -> D -> A
def prev_matrimoiety(c):
    return chr((ord(c) - ord('A') - 1) % 4 + ord('A')) # Get the next character in the order: A -> D -> C -> B -> A

class Warlpiri_Subsection(object):
    def __init__(self, semipatrimoiety):
        self.SemiPatrimoiety = semipatrimoiety
    def get_matrimoiety(self):
        return SemiPatrimoiety_Dictionary[self.SemiPatrimoiety]
    def get_child_node(self):
        if self.SemiPatrimoiety.endswith("A"):
            return Warlpiri_Subsection(self.SemiPatrimoiety.replace("A", "B"))
        else:
            return Warlpiri_Subsection(self.SemiPatrimoiety.replace("B", "A"))
    def get_ideal_wife(self):
        child_node = self.get_child_node()
        child_matrimoiety = child_node.get_matrimoiety()
        if child_matrimoiety.startswith("M1"): # Move forward a step of matrimoiety if the node is in M1 (I.e. M1B -> M1C)
            mother_matrimoiety = (
                    child_matrimoiety[:-1] +
                    next_matrimoiety(child_matrimoiety[-1])
            )
        else: # Reverse the step of matrimoiety if the node is in M1 (I.e. M1C -> M1B)
            mother_matrimoiety = (
                    child_matrimoiety[:-1] +
                    prev_matrimoiety(child_matrimoiety[-1])
            )
        return Warlpiri_Subsection(Matrimoiety_Dictionary[mother_matrimoiety])
    def get_parents(self): # Returns [FatherNode, MotherNode]
        father_node = self.get_child_node() # The father's node will always be the same Patrimoiety with an inverse subsection (the same as child nodes)
        mother_node = father_node.get_ideal_wife()
        return [father_node, mother_node]

if __name__ == "__main__":
    testNode = Warlpiri_Subsection("P1A")

    # Lets Create a lookup table to save resources. one for ideal wife one for child
    # print("P1A -> ", Warlpiri_Subsection("P1A").get_child_node().SemiPatrimoiety)
    # print("P1B -> ", Warlpiri_Subsection("P1B").get_child_node().SemiPatrimoiety)
    # print("P2A -> ", Warlpiri_Subsection("P2A").get_child_node().SemiPatrimoiety)
    # print("P2B -> ", Warlpiri_Subsection("P2B").get_child_node().SemiPatrimoiety)
    # print("P3A -> ", Warlpiri_Subsection("P3A").get_child_node().SemiPatrimoiety)
    # print("P3B -> ", Warlpiri_Subsection("P3B").get_child_node().SemiPatrimoiety)
    # print("P4A -> ", Warlpiri_Subsection("P4A").get_child_node().SemiPatrimoiety)
    # print("P4B -> ", Warlpiri_Subsection("P4B").get_child_node().SemiPatrimoiety)
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
    print("P1A -> ", Warlpiri_Subsection("P1A").get_ideal_wife().SemiPatrimoiety)
    print("P1B -> ", Warlpiri_Subsection("P1B").get_ideal_wife().SemiPatrimoiety)
    print("P2A -> ", Warlpiri_Subsection("P2A").get_ideal_wife().SemiPatrimoiety)
    print("P2B -> ", Warlpiri_Subsection("P2B").get_ideal_wife().SemiPatrimoiety)
    print("P3A -> ", Warlpiri_Subsection("P3A").get_ideal_wife().SemiPatrimoiety)
    print("P3B -> ", Warlpiri_Subsection("P3B").get_ideal_wife().SemiPatrimoiety)
    print("P4A -> ", Warlpiri_Subsection("P4A").get_ideal_wife().SemiPatrimoiety)
    print("P4B -> ", Warlpiri_Subsection("P4B").get_ideal_wife().SemiPatrimoiety)
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
    # Fathers = P1A, P1B, P2B, P3B
