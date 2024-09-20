import numpy as np
from scipy.stats import binom
from MDP import *

# Define parameters.
M = 5  # Maximum spare parts the repairman can carry
N = 5  # Number of towns
num_states = (M + 1) * N  # States 0 to M
num_actions = 2  # Actions: 0 (do not replenish), 1 (replenish to M)
max_demand = 2  # Maximum demand per town

# Towns data
p = [0.5, 0.25, 0.35, 0.3, 0.5]  # Demand probabilities for towns 1 to 5
c = [60, 30, 50, 25, 100]  # Replenishment costs cj for towns 1 to 5
K = [200, 200, 200, 200, 200]  # Extra visit costs Kj for towns 1 to 5

horizon = 5  # Number of days

# Initialize transitions and rewards lists
transitions = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions))

for state in range(num_states):
    s = state % (M + 1)  # Spare parts in the repairman's inventory (0 to M)
    j = state // (M + 1)  # Current town (0 to N-1)
    j_prime = j+1 if j < N-1 else 0
    pj = p[j_prime]
    demand_probs = [binom.pmf(k, 2, pj) for k in range(max_demand + 1)]  # Demands 0, 1, 2

    for a in range(num_actions):
        # Initialize expected cost and transition probabilities
        expected_cost = 0.0
        trans_probs = np.zeros(num_states)

        for d in range(max_demand + 1):  # Demands 0 to 2
            prob_d = demand_probs[d]
            spare_parts_used = min(s, d)
            extra_visit = 1 if d > s else 0

            # Spare parts after demand
            s_after_demand = s - spare_parts_used

            # Spare parts at the start of next day
            if a == 1:  # Replenish to full capacity
                s_next = M
            else:  # Do not replenish
                s_next = s_after_demand

            # Next town
            j_next = (j + 1) % N  
            state_next = s_next + j_next * (M + 1)

            # Transition probability
            trans_probs[state_next] += prob_d

            # Immediate cost
            extra_visit_cost = K[j] if extra_visit else 0
            replenishment_cost = c[j] if a == 1 else 0

            # Immediate cost (we are minimizing cost)
            expected_cost += prob_d * (extra_visit_cost + replenishment_cost)

        # Normalize transition probabilities (ensure they sum to 1)
        total_prob = np.sum(trans_probs)
        if total_prob > 0:
            trans_probs /= total_prob
        else:
            # This should not happen; if it does, assign equal probabilities
            trans_probs[:] = 1.0 / num_states

        # Assign the transition probabilities
        transitions[state, a, :] = trans_probs

        # Assign the expected reward
        rewards[state, a] = -expected_cost

# Initialize and solve the MDP
mdp = FiniteHorizonMDP(num_states, num_actions, transitions, rewards, horizon)
V, policy, dV = mdp.solve()

# Print the optimal policy
print("Value:")
print(V)
print("\nPolicy:")
print(policy)
print("\nValue per step:")
print(dV)


