import numpy as np
from MDP import *

# Define parameters.
num_states = 5 # States 1 to 4 and 0 is the obsorbing state
num_actions = 2  # Actions: 0 (do not exit), 1 (exit)

# Reward remaining in the system
r = [i for i in range(1, num_states+1)]

# Transition probabilities
P = [[0.3, 0.4, 0.2, 0.1],
     [0.2, 0.3, 0.5, 0.0],
     [0.1, 0.0, 0.8, 0.1],
     [0.4, 0.0, 0.0, 0.6]]

# Initialize transitions and rewards lists
transitions = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions))


def solve(R):
    for state in range(num_states):
        if state == 0:
            # From state 0, any action leads to state 0 with probability 1.0
            transitions[state, 0:2, 0] = 1.0
            rewards[state, 0:2] = 0
        else:
            # From states 1 to 4, action 0 leads to states 1 to 4 with probabilities defined by P
            transitions[state, 0, 1:5] = P[state-1]
            rewards[state, 0] = r[state-1]
            # From states 1 to 4, action 1 leads to state 0 with probability 1.0
            transitions[state, 1, 0] = 1.0
            rewards[state, 1] = R

    mdp = InfiniteHorizonMDP(num_states, num_actions, transitions, rewards, discount_factor=0.9)
    V, policy = mdp.value_iteration_discount(silence=True)
    return V, policy

# Find the minimum R that makes the agent exit the system at state 2
Rh = 1000
Rl = 20
iteration = 0
# Bisection method
while abs(Rh - Rl) > 1e-6:
    iteration += 1
    R = (Rh + Rl) / 2
    V, policy = solve(R)
    print(f"Iteration {iteration}: R = {R}, V = {V}")
    if policy[2] == 1:
        Rh = R
    else:
        Rl = R

print(f"R = {R}")
print(f"V = {V}")
print(f"policy = {policy}")

    