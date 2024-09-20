import numpy as np
from MDP import *

# Define states and actions.
num_states = 6
states = np.arange(num_states)
actions = np.array([0, 1])  # Actions: 0 and 1
num_actions = len(actions)
P = 25 

# Define transition probabilities.
Q = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0],   
    [0.0, 0.8, 0.1, 0.05, 0.05], 
    [0.0, 0.0, 0.7, 0.1, 0.2],   
    [0.0, 0.0, 0.0, 0.5, 0.5],   
])

# Initialize the transitions array.
transitions = np.zeros((num_states, num_actions, num_states))

for s in states:
    if s == 0:
        # From state 0, any action leads to state 1 with probability 1.0
        for a in range(num_actions):
            transitions[s, a, 1] = 1.0
    elif s == 5:
        # From state 5, any action leads to state 0 with probability 1.0
        for a in range(num_actions):
            transitions[s, a, 0] = 1.0
    else:
        for a in range(num_actions):
            if a == 0:
                # Action 0: transitions defined by Q
                transitions[s, a, 1:6] = Q[s-1]
            else:
                # Action 1: transitions to state 1 with probability 1.0
                transitions[s, a, 1] = 1.0

# Define rewards.
rewards = np.zeros((num_states, num_actions))
c_p = [0, 7, 7, 5]  # Rewards for action 1 in states 1 to 4

for s in states:
    for a in range(num_actions):
        if s == 0:
            rewards[s, a] = 0
        elif s == 5:
            rewards[s, a] = -10
        elif a == 0:
            rewards[s, a] = P
        elif a == 1:
            rewards[s, a] = -c_p[s-1]

# Set the time horizon.
horizon = 7

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
V_v, policy_v = ihmdp.value_iteration()
print("Optimal Value:")
print(V_v)
print("Optimal Policy:")
print(policy_v)