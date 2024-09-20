import numpy as np

class FiniteHorizonMDP:
    def __init__(self, num_states, num_actions, transitions, rewards, horizon):
        """
        Initialize the finite horizon MDP.

        :param num_states: Number of possible states.
        :param num_actions: Number of possible actions.
        :param transitions: 3D NumPy array transitions[s, a, s'] = P(s'|s,a).
        :param rewards: 2D NumPy array rewards[s, a] = R(s, a).
        :param horizon: The finite time horizon T.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transitions = transitions  # Shape: (num_states, num_actions, num_states)
        self.rewards = rewards          # Shape: (num_states, num_actions)
        self.horizon = horizon

    def solve(self):
        """
        Solves the finite horizon MDP using dynamic programming.
        Returns: 
        - V: optimal value function 
        - policy: for each time step
        - dV: value per step
        """
        V = np.zeros((self.horizon + 1, self.num_states))
        policy = np.zeros((self.horizon, self.num_states), dtype=int)
        dV = np.zeros((self.horizon + 1, self.num_states))

        # Backward induction.
        for t in range(1, self.horizon):
            for s in range(self.num_states):
                action_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    expected_value = np.sum(
                        self.transitions[s, a, :] * V[t - 1, :]
                    )
                    action_values[a] = self.rewards[s, a] + expected_value
                # Since we are maximizing profit, we use argmax.
                best_action = np.argmax(action_values)
                V[t, s] = action_values[best_action]
                policy[t, s] = best_action
                dV[t, s] = V[t, s] - V[t-1, s]
        return V, policy, dV
    
    

class InfiniteHorizonMDP:
    def __init__(self, num_states, num_actions, transitions, rewards, gamma=1.0, epsilon=1e-2, discount_factor=0.9):
        """
        Initialize the infinite horizon MDP.

        :param num_states: Number of possible states.
        :param num_actions: Number of possible actions.
        :param transitions: 3D NumPy array transitions[s, a, s'] = P(s'|s,a).
        :param rewards: 2D NumPy array rewards[s, a] = Expected immediate reward.
        :param gamma: Pertubation factor (0 < gamma <= 1).
        :param epsilon: Convergence threshold for value iteration.
        :param discount_factor: Discount factor.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transitions = transitions  # Shape: (num_states, num_actions, num_states)
        self.rewards = rewards          # Shape: (num_states, num_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.discount = discount_factor

    def value_iteration(self):
        """
        Solves the infinite horizon MDP using value iteration.
        Returns:
        - V: Optimal value function.
        - policy: Optimal policy.
        """
        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)
        iteration = 0
        transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        # prob pertubation
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_next in range(self.num_states):
                    if s_next != s:
                        transitions[s, a, s_next] = self.transitions[s, a, s_next] * self.gamma
                    else:
                        transitions[s, a, s_next] = self.transitions[s, a, s_next] * self.gamma + (1 - self.gamma)
        
        while True:
            delta_U = -np.inf
            delta_L = np.inf
            V_new = np.zeros_like(V)
            for s in range(self.num_states):
                action_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    expected_value = np.sum(
                        transitions[s, a, :] * V
                    )
                    action_values[a] = self.rewards[s, a] + expected_value
                best_action_value = np.max(action_values)
                V_new[s] = best_action_value
                best_action = np.argmax(action_values)
                policy[s] = best_action
                delta_U = max(delta_U, abs(V_new[s] - V[s]))
                delta_L = min(delta_L, abs(V_new[s] - V[s]))

            V = V_new

            iteration += 1
            delta = (delta_U - delta_L) / (delta_L + self.epsilon)

            print(f"Iteration {iteration}: delta = {delta}")
            print(f'V = {V}')

            if delta < self.epsilon:
                break

        return V, policy
    

    def policy_iteration(self):
        """
        Perform policy iteration to find the optimal policy.

        Returns:
            policy: NumPy array of shape (num_states,) representing the optimal action for each state.
            V: NumPy array of shape (num_states,) representing the value function for the optimal policy.
        """
        # Initialize policy arbitrarily
        policy = np.zeros(self.num_states, dtype=int)  # Initial policy: choose action 0 for all states

        is_policy_stable = False

        while not is_policy_stable:
            # Policy Evaluation
            P_pi = np.zeros((self.num_states, self.num_states))
            R_pi = np.zeros(self.num_states)

            for s in range(self.num_states):
                a = policy[s]
                P_pi[s, :] = self.transitions[s, a, :]
                R_pi[s] = self.rewards[s, a]

            # Solve V = R_pi + P_pi * V - g => (I - P_pi) * V + g = R_pi
            s0 = 0
            N = self.num_states

            # Construct the linear system
            A = np.zeros((N + 1, N + 1))
            b = np.zeros(N + 1)

            # (I - P_pi) V - e * g = R_pi
            A[:N, :N] = np.eye(N) - P_pi
            A[:N, N] = -1  # Coefficient for -g
            b[:N] = R_pi

            # Reference state constraint V[s0] = 0
            A[N, s0] = 1
            b[N] = 0

            # Solve the linear system
            solution = np.linalg.solve(A, b)
            V = solution[:N]
            g = solution[N]

            # Policy Improvement
            is_policy_stable = True
            for s in range(self.num_states):
                # Compute Q(s, a) for all actions
                Q_sa = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    Q_sa[a] = self.rewards[s, a] - g + np.dot(self.transitions[s, a, :], V)

                # Find the best action
                best_action = np.argmax(Q_sa)
                if best_action != policy[s]:
                    is_policy_stable = False
                    policy[s] = best_action

        return V, policy
    
    def value_iteration_discount(self, silence=False):
        """
        Solves the infinite horizon MDP using value iteration with discount factor.
        Returns:
        - V: Optimal value function.
        - policy: Optimal policy.
        """
        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)
        iteration = 0
        transitions = self.transitions
        
        while True:
            delta_U = -np.inf
            delta_L = np.inf
            V_new = np.zeros_like(V)
            for s in range(self.num_states):
                action_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    expected_value = np.sum(
                        transitions[s, a, :] * V
                    )
                    action_values[a] = self.rewards[s, a] + expected_value * self.discount
                best_action_value = np.max(action_values)
                V_new[s] = best_action_value
                best_action = np.argmax(action_values)
                policy[s] = best_action
                delta_U = max(delta_U, abs(V_new[s] - V[s]))
                delta_L = min(delta_L, abs(V_new[s] - V[s]))

            V = V_new

            iteration += 1
            delta = (delta_U - delta_L) / (delta_L + self.epsilon)

            if not silence:
                print(f"Iteration {iteration}: delta = {delta}")
                print(f'V = {V}')

            if delta < self.epsilon:
                break

        return V, policy



class SemiMDP:
    def __init__(self, num_states, num_actions, transitions, sojourtime, rewards, tau=1.0, epsilon=1e-2, discount_factor=0.9):
        """
        Initialize the infinite horizon MDP.

        :param num_states: Number of possible states.
        :param num_actions: Number of possible actions.
        :param transitions: 3D NumPy array transitions[s, a, s'] = P(s'|s,a).
        :param sojourtime: 2D NumPy array sojourtime[s, a] = Expected sojourtime.
        :param rewards: 2D NumPy array rewards[s, a] = Expected immediate reward.
        :param gamma: Pertubation factor (0 < gamma <= 1).
        :param epsilon: Convergence threshold for value iteration.
        :param discount_factor: Discount factor.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transitions = transitions  # Shape: (num_states, num_actions, num_states)
        self.sojourtime = sojourtime  # Shape: (num_states, num_actions)
        self.rewards = rewards/sojourtime          # Shape: (num_states, num_actions)
        self.tau = tau
        self.epsilon = epsilon
        self.discount = discount_factor

    def value_iteration(self, silence=False):
        """
        Solves the infinite horizon MDP using value iteration.
        Returns:
        - V: Optimal value function.
        - policy: Optimal policy.
        """
        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)
        iteration = 0
        transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        # prob pertubation
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_next in range(self.num_states):
                    if s_next != s:
                        transitions[s, a, s_next] = self.transitions[s, a, s_next] * self.tau/self.sojourtime[s, a]
                    else:
                        transitions[s, a, s_next] = self.transitions[s, a, s_next] * self.tau/self.sojourtime[s, a] + (1 - self.tau/self.sojourtime[s, a])
        
        while True:
            delta_U = -np.inf
            delta_L = np.inf
            V_new = np.zeros_like(V)
            for s in range(self.num_states):
                action_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    expected_value = np.sum(
                        transitions[s, a, :] * V
                    )
                    action_values[a] = self.rewards[s, a] + expected_value
                best_action_value = np.max(action_values)
                V_new[s] = best_action_value
                best_action = np.argmax(action_values)
                policy[s] = best_action
                delta_U = max(delta_U, abs(V_new[s] - V[s]))
                delta_L = min(delta_L, abs(V_new[s] - V[s]))

            V = V_new

            iteration += 1
            delta = (delta_U - delta_L) / (delta_L + self.epsilon)
            avg_reward = delta_U

            if not silence:
                print(f"Iteration {iteration}: delta = {delta}")
                print(f'V = {V}')

            if delta < self.epsilon:
                break

        return V, policy, avg_reward
