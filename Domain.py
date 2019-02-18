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
        self.w = rdm.uniform(0, 1)  # Noise of the current time
        self.maxReward = np.max(board)

    def getGamma (self):
        return self.gamma

    def getBeta(self):
        return self.beta

    def getBoard(self):
        return self.board

    def getShape(self):
        return self.x_max, self.y_max

    def drawNoise(self):
        self.w = rdm.uniform(0, 1)

    def getMaxReward(self):
        return self.maxReward

    def deterministicMove(self, position, move):
        assert move in Domain.VALID_ACTIONS, "Please use a valid move"
        x_move, y_move = move
        x_pos, y_pos = position
        x_pos = min(max(x_pos + x_move, 0), self.x_max-1)
        y_pos = min(max(y_pos + y_move, 0), self.y_max-1)
        return x_pos, y_pos

    def move(self, position, move):
        assert move in Domain.VALID_ACTIONS, "Please use a valid move"
        x_pos, y_pos = (0, 0)
        if self.w <= 1 - self.beta:
            x_pos, y_pos = self.deterministicMove(position, move)
        return x_pos, y_pos

    def moveResult(self, initialPosition, move):
        res = [(self.deterministicMove(initialPosition, move), 1-self.beta)]
        if self.beta != 0:
            res.append(((0, 0), self.beta))
        return res

    def computeProbability(self,initialPosition, move, finalState):
        if finalState == (0, 0):
            return self.beta
        if self.deterministicMove(initialPosition, move) == finalState:
            return 1-self.beta
        return 0

    def deterministicReward(self, position, move):
        x, y = self.deterministicMove(position, move)
        return self.board[y][x]

    def reward(self, position, move):
        x, y = self.move(position, move)
        return self.board[y][x]

    def expectedReward(self, position, move):
        return ((1 - self.beta)*self.deterministicReward(position, move)) + (self.beta * self.board[0][0])
