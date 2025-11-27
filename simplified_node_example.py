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

class Node(object):
    def __init__(self, semipatrimoiety):
        self.SemiPatrimoiety = semipatrimoiety
    def get_matrimoiety(self):
        return SemiPatrimoiety_Dictionary[self.SemiPatrimoiety]
    def get_child_node(self):
        if self.SemiPatrimoiety.endswith("A"):
            return Node(self.SemiPatrimoiety.replace("A", "B"))
        else:
            return Node(self.SemiPatrimoiety.replace("B", "A"))
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
        return Node(Matrimoiety_Dictionary[mother_matrimoiety])
    def get_parents(self): # Returns [FatherNode, MotherNode]
        father_node = self.get_child_node() # The father's node will always be the same Patrimoiety with an inverse subsection (the same as child nodes)
        mother_node = father_node.get_ideal_wife()
        return [father_node, mother_node]


testNode = Node("P1A")
print(testNode.get_parents()[0].SemiPatrimoiety)
print(testNode.get_parents()[1].SemiPatrimoiety)
print(testNode.get_ideal_wife().SemiPatrimoiety)
print(testNode.get_parents()[1].get_parents()[0].get_parents()[1].get_parents()[0].SemiPatrimoiety) # MFMF - Should be equal to the original node