import numpy as np
from MDP import *

# Define states and actions.
B = 5
num_states = B+1 # Queue length B
num_actions = 5 # Number of speeds K

# Data
speed = np.arange(1, num_actions+1)
lambd = 10
mu = 1
operation_cost = speed * 2
lost_cost = 100

# Initialize transitions and rewards lists
transitions = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions))
sojourntime = np.zeros((num_states, num_actions))

for s in range(num_states):
    for a in range(num_actions):
        # Compute the probability of transitioning to each next state.
        if 1 <= s < num_states - 1:
            transitions[s, a, s+1] = lambd/(lambd + mu * speed[a])
            transitions[s, a, s-1] = mu * speed[a]/(lambd + mu * speed[a])
            sojourntime[s, a] = 1/(lambd + mu * speed[a])
        elif s == 0:
            transitions[s, a, s+1] = 1
            sojourntime[s, a] = 1/lambd
        else:
            transitions[s, a, s] = lambd/(lambd + mu * speed[a])
            transitions[s, a, s-1] = mu * speed[a]/(lambd + mu * speed[a])
            sojourntime[s, a] = 1/(lambd + mu * speed[a])

        

        rewards[s, a] = -operation_cost[a] * sojourntime[s, a] - lost_cost * lambd * (s == B) * sojourntime[s, a]


# Initialize and solve the MDP.
tau = min(sojourntime.flatten()) * 0.9
smdp = SemiMDP(num_states, num_actions, transitions, sojourntime, rewards, tau=tau)
V, policy, avg_reward = smdp.value_iteration()
print("Value:")
print(V)
print("\nPolicy:")
print(policy)
print("\nTau:")
print(sojourntime)
print(f"\n Rewards: {rewards}")
print(f"\n Speed: {speed}")
print(f"\n Operation Cost: {operation_cost}")
for a in range(num_actions):
    print(f"\n Action: {transitions[:, a, :]}")  

print(f"\n Average Reward: {avg_reward}")