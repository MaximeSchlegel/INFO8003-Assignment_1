# Deterministic = Stochastic with Beta=0

import random as rdm
import numpy as np


class Domain:
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    VALID_ACTIONS = [UP, RIGHT, DOWN, LEFT]

    def __init__(self, board, beta=0, gamma=0.99):
        # By default a deterministic domain is create
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Probability of returning to the origin
        self.board = board  # give the reward obtain by landing on a cell
        self.y_max, self.x_max = board.shape
        self.noise = rdm.uniform(0, 1)  # Noise of the current time
        self.max_reward = np.max(board)

    def get_gamma(self):
        return self.gamma

    def get_shape(self):
        return self.x_max, self.y_max

    def update_noise(self):
        self.noise = rdm.uniform(0, 1)

    def get_max_reward(self):
        return self.max_reward

    def deterministic_move(self, position, move):
        assert move in Domain.VALID_ACTIONS, "Please use a valid move"
        x_move, y_move = move
        x_pos, y_pos = position
        x_pos = min(max(x_pos + x_move, 0), self.x_max-1)
        y_pos = min(max(y_pos + y_move, 0), self.y_max-1)
        return x_pos, y_pos

    def deterministic_reward(self, position, move):
        x, y = self.deterministic_move(position, move)
        return self.board[y][x]

    def move(self, position, move):
        assert move in Domain.VALID_ACTIONS, "Please use a valid move"
        x_pos, y_pos = (0, 0)
        if self.noise <= 1 - self.beta:
            x_pos, y_pos = self.deterministic_move(position, move)
        return x_pos, y_pos

    def reward(self, position, move):
        x, y = self.move(position, move)
        return self.board[y][x]

    def compute_move_probability(self, initial_position, move, final_state):
        if final_state == (0, 0):
            return self.beta
        if self.deterministic_move(initial_position, move) == final_state:
            return 1-self.beta
        return 0

    def expected_move(self, initial_position, move):
        res = [(self.deterministic_move(initial_position, move), 1 - self.beta)]
        if self.beta != 0:
            res.append(((0, 0), self.beta))
        return res

    def expected_reward(self, position, move):
        return ((1 - self.beta) * self.deterministic_reward(position, move)) + (self.beta * self.board[0][0])
