# Warlpiri-based Coarse-Grained Genetic Algorithm (WCGGA)
- WCGGA is not a co-evolutionary algorithm. It is more closely related to the Coarse-Grained Genetic Algorithm.
  - **Competitive** Co-Evolutionism is incompatible with the Knapsack Problem
- **Colaborative** Co-Evolutionism would consist of each individual group working on a subset of the problem (e.g. Group 1 works on items [0-5], Group 2 on items [6-10]...)
  - This is not possible for the Warlpiri implementation as in Colaborative algorithms, individuals cannot migrate groups (by definition).
## The Standard Coarse-Grained Genetic Algorithm (CGGA)
The standard Coarse Grained algorithm is set up with **_N_** groups and the flow is structured:
- Parents are chosen from within their group (I.e. Group 1 Male paired with Group 1 Female)
  - Every **_X_** generations, a few individuals migrate in a simple ring

        (Group 1 -> Group 2 -> ... -> Group N -> Group 1)
The benefit derrived from CGG Algorithms is it's ability for each group to be processed in parrallel. This cannot be achieved with WCGGA as it's required by Warlpiri Kinship rules to migrate children every generation.

## WCGGA Algorithm Flow
### Initialization
- 8 Groups are defined in size by the ***floor(POPULATION_SIZE / 8)***
- Individuals in each group are initialized using a greedy value/weight ratio.
- Initialization obides by a ratio (10%), where all other individuals are randomly initialized.
  - Additionally, individuals initialized with the greedy algorithm have a 10% chance for every item to skip (ensuring genetic diversity)
### GA Main Loop
- For every iteration in the GA loop, each population is iterated over once.
  - For every population, each individual in the population is iterated over once.
    - For every individual (chosen by a tournament of two random individuals);
      - An individual from their pre-defined marrigage group is chosen as their partner (also through a 2 person tournament selection)
      - Both parents are cloned and are considered the two offspring from the parents
      - The children (cloned parents) then have a percentage chance of performing **TWO-POINT** crossover (`P_CROSSOVER`) with each other
      - Each child then has a percentage chance of performing **BIT FLIP** mutation (`P_MUTATION`)
      - Once crossover and mutation has potentially been applied, each child's value is evaluated
        - If the child is overweight, the highest weight/value ratio items are removed until the child is under or equal to the maximum weight.
      - These children are then added to a list of kids that will be added for the next generation in their predefined group (based on Warlpiri system)
  - Once every population has been cycled through, the list of kids designated for each population is sorted by fitness.
  - The lowest **X** children are removed and replaced with the **X** highest fitness individuals from the previous generation (`ELITE_SIZE`)
