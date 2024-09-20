import numpy as np
from scipy.stats import binom
from MDP import *

# Define parameters.
num_of_states = 6  # Possible inventory levels from 0 to 5.
num_of_actions = 6  # Possible order quantities from 0 to 5.
states = np.arange(num_of_states)
actions = np.arange(num_of_actions)
B = 5  # Maximum inventory level and maximum demand.
K = 4  # Fixed ordering cost
c = 2  # Unit ordering cost
h = 1  # Holding cost per unit
p = 8  # Price per unit sold (revenue)
b = 2  # Lost sale cost per unit
demand_prob = 0.6  # Probability parameter for binomial distribution

# Set the time horizon.
horizon = 11

# Define transition probabilities and rewards.
num_states = num_of_states
num_actions = num_of_actions

transitions = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions))

# Precompute demand probabilities.
demand_probs = np.array([binom.pmf(k, B, demand_prob) for k in range(B + 1)])

for s in states:
    for a in actions:
        # Compute new inventory level after ordering.
        inventory_level = min(B, s + a)
        max_possible_sales = min(B, inventory_level)  # Can't sell more than demand or inventory

        # Compute the probability of transitioning to each next state.
        for next_s in states:
            for demand in range(B + 1):
                # The next state is determined by the inventory level minus demand, bounded between 0 and B.
                next_inventory = max(0, inventory_level - demand)
                if next_inventory == next_s:
                    prob = demand_probs[demand]
                    transitions[s, a, next_s] += prob

        # Normalize probabilities to ensure they sum to 1.
        transitions[s, a, :] /= transitions[s, a, :].sum()

        # Compute expected immediate reward for state s and action a.
        # Rewards include revenue from sales minus costs.
        ordering_cost = K * (a > 0) + c * a
        expected_holding_cost = 0
        expected_lost_sale_cost = 0
        expected_revenue = 0

        for demand in range(B + 1):
            prob = demand_probs[demand]
            sales = min(inventory_level, demand)
            leftover_inventory = inventory_level - sales
            unmet_demand = max(0, demand - inventory_level)

            holding_cost = h * leftover_inventory
            lost_sale_cost = b * unmet_demand
            revenue = p * sales

            expected_holding_cost += prob * holding_cost
            expected_lost_sale_cost += prob * lost_sale_cost
            expected_revenue += prob * revenue

        total_expected_reward = (
            expected_revenue
            - ordering_cost
            - expected_holding_cost
            - expected_lost_sale_cost
        )

        rewards[s, a] = total_expected_reward

# Initialize and solve the MDP.
mdp = FiniteHorizonMDP(num_states, num_actions, transitions, rewards, horizon)
V, policy, dV = mdp.solve()


# Print the results.

for t in range(horizon):
    print(f"Time {t}:")
    print("Values:", V[t])
    print("Policy:", policy[t])
    print("Value per step:", dV[t])


ihmdp = InfiniteHorizonMDP(num_states, num_actions, transitions, rewards)
V_v, policy_v = ihmdp.value_iteration_discount()

# Print the results.
print("Optimal Value Function:")
print(V_v)
print("Optimal Policy:")
print(policy_v)

V_p, policy_p = ihmdp.policy_iteration()
print("Optimal Value Function:")
print(V_p)
print("Optimal Policy:")
print(policy_p)