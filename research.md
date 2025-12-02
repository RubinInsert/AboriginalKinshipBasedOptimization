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

