import numpy as np
import random as rdm
from src.Agent import Agent
from src.utils import decompose_trajectory


class QAgent(Agent):

    def __init__(self, domain, policy=None,  alpha=0.05, replay=False, max_sample=None, trajectory=None):
        if policy:
            super().__init__(domain, policy)
        else:
            super().__init__(domain, self.policy_mu)
        self.alpha = alpha
        self.replay = replay
        self.max_sample = max_sample
        self.history = []
        self.max_reward = 0
        self.action_state_matrix = self.action_state_matrix[0]
        if trajectory:
            self.analyze_trajectory(trajectory)

    def get_action_state_matrix(self, n=None):
        return self.action_state_matrix

    def get_history(self):
        return self.history

    def get_max_reward(self):
        return self.max_reward

    def save_and_analyze_one_step_transition(self, initial_position, move, reward, final_position, save=True):
        if save:
            self.history.append((initial_position, move, reward, final_position))
        if reward > self.max_reward:
            self.max_reward = reward
        previous_q_value = self.action_state_matrix[initial_position[1]][initial_position[0]][move]
        final_q = self.action_state_matrix[final_position[1]][final_position[0]]
        q_update = ((1 - self.alpha) * previous_q_value) + self.alpha * (reward + self.domain.get_gamma() * final_q[max(final_q, key=final_q.get)])
        self.action_state_matrix[initial_position[1]][initial_position[0]][move] = q_update

    def experience_replay(self, nb_sample):
        ht = rdm.choices(self.history, k=nb_sample)
        for osst in ht:
            self.save_and_analyze_one_step_transition(osst[0], osst[1], osst[2], osst[3], False)

    def analyze_trajectory(self, trajectory):
        ht = decompose_trajectory(trajectory)
        self.history += ht
        for osst in ht:
            self.save_and_analyze_one_step_transition(osst[0], osst[1], osst[2], osst[3])
            if self.experience_replay:
                if self.max_sample is not None:
                    self.experience_replay(min(self.max_sample, len(self.history)))
                else:
                    for osst in self.history:
                        self.save_and_analyze_one_step_transition(osst[0], osst[1], osst[2], osst[3], False)

    def explore_and_analyze(self, n=1000):
        seen = set()

        def explore_policy(domain, position):
            to_take = []
            for action in domain.VALID_ACTIONS:
                if not (position, action) in seen:
                    to_take.append(action)
            if to_take:
                action = rdm.choice(to_take)
                seen.add((position, action))
                return action
            return rdm.choice(domain.VALID_ACTIONS)

        old_policy = self.policy
        self.policy = explore_policy
        x_max, y_max = self.domain.get_shape()
        trajectory = self.play(n, (rdm.randint(0, x_max-1), rdm.randint(0, y_max-1)))
        self.analyze_trajectory(trajectory)
        self.policy = old_policy

    def expected_return_error(self, n):
        return (pow(self.domain.get_gamma(), n) * self.get_max_reward()) / (1 - self.domain.get_gamma())

    def expected_return_display(self, n=None, precision=0):
        print("Matrix of the Expected Return :")
        if n is None:
            a = 0
            b = len(self.expected_return_matrix)
        elif n == -1:
            a = len(self.expected_return_matrix) - 1
            b = len(self.expected_return_matrix)
        else:
            a = n
            b = n + 1
        for i in range(a, b):
                print("N = " + str(i))
                with np.printoptions(precision=precision):
                    print(self.expected_return_matrix[i], "\n")

    def action_state_compute(self, n, state=None, erase=False):
        pass

    def action_state_error(self, n):
        pass

    def action_state_approximate(self, error=0.0001, erase=False):
        pass

    def action_state_display(self, n=None, precision=0):
        print("Matrix of the Action-State :")
        str_buffer = "["
        for j in range(len(self.action_state_matrix)):
            str_buffer += "[" if j == 0 else " ["
            for i in range(len(self.action_state_matrix[j])):
                str_buffer += "{" if i == 0 else "  {"
                for k in self.action_state_matrix[j][i].keys():
                    str_buffer += str(k) + ": " + str(np.floor(self.action_state_matrix[j][i][k] * (10 ** precision)) / (10 ** precision))
                    if k != list(self.action_state_matrix[j][i].keys())[-1]:
                        str_buffer += "  "
                str_buffer += "}\n" if i != len(self.action_state_matrix[j]) - 1 else "}"
            str_buffer += "]\n"
        print(str_buffer + "]", "\n")

    def policy_mu(self, domain, position):
        return max(self.action_state_matrix[position[1]][position[0]], key=self.action_state_matrix[position[1]][position[0]].get)

    def policy_get_mu(self, error=0.0001, use=False):
        if use:
            self.policy = self.policy_mu
        return self.policy_mu

