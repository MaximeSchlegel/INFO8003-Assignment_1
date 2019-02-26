import numpy as np
import random as rdm
from src.Agent import Agent
from src.MDPDomain import MDPDomain


class MDPAgent(Agent):

    def __init__(self, domain, policy, nb_moves=1000, history=None):
        super().__init__(domain, policy)
        self.emulated_domain = MDPDomain()
        if history is None:
            self.explore_and_analyze(nb_moves)
        else:
            self.emulated_domain.analyze_trajectory(history)

    def get_emulated_domain(self):
        return self.emulated_domain

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
        history = self.play(n, (rdm.randint(0, x_max), rdm.randint(0, y_max)))
        self.emulated_domain.analyze_trajectory(history)
        self.policy = old_policy

    def expected_return_compute(self, n, state=None, erase=False):
        x_max, y_max = self.domain.get_shape()
        if erase:
            self.expected_return_matrix = [np.array([[0. for _ in range(x_max)] for _ in range(y_max)])]
        if len(self.expected_return_matrix) > n:
            if state is not None:
                return self.expected_return_matrix[n][state[0]][state[1]]
            return self.expected_return_matrix[n]
        for it in range(n+1-len(self.expected_return_matrix)):
            previous_matrix = self.expected_return_matrix[it - 1]
            self.expected_return_matrix.append(np.array([[0. for _ in range(x_max)] for _ in range(y_max)]))
            for i in range(x_max):
                for j in range(y_max):
                    move = self.policy(self.domain, (i, j))
                    self.expected_return_matrix[-1][j][i] = self.emulated_domain.expected_reward((i, j), move)
                    for new_position, probability in self.emulated_domain.expected_move((i, j), move):
                        self.expected_return_matrix[-1][j][i] += self.domain.get_gamma() * probability * previous_matrix[new_position[1]][new_position[0]]
        if state is not None:
            return self.expected_return_matrix[-1][state[0]][state[1]]
        return self.expected_return_matrix[-1]

    def expected_return_error(self, n):
        return (pow(self.domain.get_gamma(), n) * self.emulated_domain.get_max_reward()) / (1 - self.domain.get_gamma())

    def action_state_compute(self, n, state=None, erase=False):
        x_max, y_max = self.domain.get_shape()
        if erase:
            self.action_state_matrix = [np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for _ in range(x_max)] for _ in range(y_max)])]
        if len(self.action_state_matrix) > n:
            if state is not None:
                return self.action_state_matrix[n][state[0]][state[1]]
            return self.action_state_matrix[n]
        for it in range(n+1-len(self.action_state_matrix)):
            previous_matrix = self.action_state_matrix[it - 1]
            self.action_state_matrix.append(np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for _ in range(x_max)] for _ in range(y_max)]))
            for i in range(x_max):
                for j in range(y_max):
                    for move in self.domain.VALID_ACTIONS:
                        self.action_state_matrix[-1][j][i][move] += self.emulated_domain.expected_reward((i, j), move)
                        for newPosition, probability in self.emulated_domain.expected_move((i, j), move):
                            self.action_state_matrix[-1][j][i][move] += self.domain.get_gamma() * probability * max(previous_matrix[newPosition[1]][newPosition[0]].values())
        if state is not None:
            return self.action_state_matrix[-1][state[0]][state[1]]
        return self.action_state_matrix[-1]

    def action_state_error(self, n):
        return (2 * pow(self.domain.get_gamma(), n) * self.emulated_domain.get_max_reward()) / (pow(1 - self.domain.get_gamma(), 2))
