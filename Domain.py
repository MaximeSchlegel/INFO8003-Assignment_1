#Deterministic = Stohastic with B=0

import random as rdm


class Domain:
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    VALID_ACTIONS = [UP, RIGHT, DOWN, LEFT]

    def __init__(self, board, beta=0, gamma=0.99):
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Probability of returning to the origin
        self.board = board  # give the reward obtain by landing on a cell
        self.y_max, self.x_max = board.shape
        self.w = rdm.random()  # Noise of the current time

    def drawNoise(self):
        self.w = rdm.uniform(0, 1)

    def deterministicMove(self, position, move):
        """
        return the position of the player starting in the position given and excuting this move
        """
        x_move, y_move = move
        x_pos, y_pos = position
        x_pos = min(max(x_pos + x_move, 0), self.x_max-1)
        y_pos = min(max(y_pos + y_move, 0), self.y_max-1)
        return x_pos, y_pos

    def move(self, position, move):
        x_pos, y_pos = (0, 0)
        if self.w <= 1 - self.beta:
            x_pos, y_pos = self.deterministicMove(position, move)
        return x_pos, y_pos

    def deterministicReward(self, position, move):
        """
        Return the reward that the player will get if he make the move given
        """
        x, y = self.deterministicMove(position, move)
        return self.board[y][x]

    def reward(self, position, move):
        x, y = self.move(position, move)
        return self.board[y][x]

    def expectedReaward(self, position, move):
        return ((1 - self.beta)*self.deterministicReward(position, move)) + (self.beta * self.board[0][0])
