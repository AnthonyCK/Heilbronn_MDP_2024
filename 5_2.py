import numpy as np
from itertools import product
from MDP import *

# Warehouse capacity
K = 4

# Rates
mu = 1      # Manufacturing rate
lambd = 1   # Demand arrival rate

# Rewards and costs
r = 200        # Revenue per satisfied demand
h = 2        # Holding cost per item per time unit
c = 100        # Operating cost per time unit when machine is on
s_cost = 50   # Switching on cost

# States: (inventory_level, machine_status)
# inventory_level: 0 to K
# machine_status: 0 (Off), 1 (On)

inventory_levels = list(range(K + 1))
machine_statuses = [0, 1]
all_states = list(product(inventory_levels, machine_statuses))
num_states = len(all_states)

# Mapping from state tuple to index and vice versa
state_to_index = {state: idx for idx, state in enumerate(all_states)}
index_to_state = {idx: state for idx, state in enumerate(all_states)}

# Actions:
num_actions = 2

transitions = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions))
sojourntime = np.zeros((num_states, num_actions))

for s_idx, (inv, status) in enumerate(all_states):
    for a in range(num_actions):
        # Determine the action effect
        if a == 0:
            # Action 0: Keep current status
            new_status = status
            switching_cost = 0.0
        else:
            # Action 1: Toggle status
            new_status = 1 - status
            switching_cost = s_cost if new_status == 1 else 0.0

        # Define the event rates based on new_status
        if new_status == 1:
            rate_total = mu + lambd
        else:
            rate_total = lambd


        sojourn_time = 1 / rate_total

        sojourntime[s_idx, a] = sojourn_time

        holding_cost = h * inv * sojourn_time
        operating_cost = c * sojourn_time if new_status == 1 else 0.0

        # Initialize expected revenue
        expected_revenue = 0.0

        # Determine possible events and transitions
        if new_status == 1:
            # Machine is On: two possible events
            # 1. Manufacturing (rate mu)
            # 2. Demand arrival (rate lambd)
            # Probability of manufacturing: mu / (mu + lambd)
            # Probability of demand: lambd / (mu + lambd)

            # Manufacturing event
            prob_manu = mu / rate_total
            # If inventory < K, add one item; else, discard
            if inv < K:
                next_inv_manu = inv + 1
            else:
                next_inv_manu = inv  # Item discarded
            next_state_manu = (next_inv_manu, new_status)
            next_idx_manu = state_to_index[next_state_manu]
            transitions[s_idx, a, next_idx_manu] += prob_manu

            # Revenue is not affected by manufacturing
            reward_manu = - (holding_cost + operating_cost + switching_cost)

            # Demand arrival event
            prob_demand = lambd / rate_total
            if inv > 0:
                next_inv_demand = inv - 1
                expected_revenue += r  # Revenue from satisfied demand
            else:
                next_inv_demand = inv  # Demand lost
            next_state_demand = (next_inv_demand, new_status)
            next_idx_demand = state_to_index[next_state_demand]
            transitions[s_idx, a, next_idx_demand] += prob_demand

            # Reward for demand event
            reward_demand = - (holding_cost + operating_cost + switching_cost) + expected_revenue

            # Assign rewards
            rewards[s_idx, a] = prob_manu * reward_manu + prob_demand * reward_demand

        else:
            # Machine is Off: only demand arrival
            prob_demand = 1.0
            if inv > 0:
                next_inv_demand = inv - 1
                expected_revenue += r  # Revenue from satisfied demand
            else:
                next_inv_demand = inv  # Demand lost
            next_state_demand = (next_inv_demand, new_status)
            next_idx_demand = state_to_index[next_state_demand]
            transitions[s_idx, a, next_idx_demand] += prob_demand

            # Reward for demand event
            reward_demand = - (holding_cost + operating_cost + switching_cost) + expected_revenue

            # Assign rewards
            rewards[s_idx, a] = prob_demand * reward_demand


# Initialize and solve the MDP.
tau = min(sojourntime.flatten()) * 0.9
smdp = SemiMDP(num_states, num_actions, transitions, sojourntime, rewards, tau=tau)
V, policy, avg_reward = smdp.value_iteration()

print("\nStates:")
print(all_states)
print(f"\n Rewards: {rewards}")
for a in range(num_actions):
    print(f"\n Action: {transitions[:, a, :]}")  

print(f"\n Average Reward: {avg_reward}")
print("\nValue:")
print(V)
print("\nPolicy:")
print(policy)
print("\nTau:")
print(sojourntime)

# Print the optimal policy
for s_idx, (inv, status) in enumerate(all_states):
    print(f"Inventory: {inv}, Status: {status}")
    action = "Toggle" if policy[s_idx] == 1 else "Keep"
    print(f"Optimal Policy: {action}")
