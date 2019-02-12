import numpy as np


class Game:

    def __init__(self, domain, policy):
        self.domain = domain
        self.policy = policy
        self.expectedReturn = [np.array([[0. for i in range(self.domain.x_max)] for j in range(self.domain.y_max)])]

    def getExpectedReturn(self):
        return self.expectedReturn

    def play(self, nbMoves, position=(0, 0)):
        position = position # the origin is on the top left of the board
        cumulatedReward = 0
        for i in range(nbMoves):
            move = self.policy(self.domain, position)
            cumulatedReward += self.domain.reward(position, move)
            position = self.domain.move(position, move)
            self.domain.drawNoise()
            print("Turn " + str(i))
            print("   Use move : " + str(move))
            print("   End the turn in positon : " + str(position))
        print("Cumulated Reward : " + str(cumulatedReward))
        print("End Position : " + str(position))
        return position, cumulatedReward

    def computeExpectedReturnMatrix(self, n):
        if len(self.expectedReturn) > n:
            return self.expectedReturn[n]

        previousExpectedReturn = self.computeExpectedReturnMatrix(n-1)

        x_max, y_max = self.domain.getShape()
        self.expectedReturn.append(np.array([[0. for i in range(x_max)] for j in range(y_max)]))

        for i in range(x_max):
            for j in range(y_max):
                move = self.policy(self.domain, (i, j))
                newPosition = self.domain.deterministicMove((i, j), move)
                self.expectedReturn[-1][j][i] = self.domain.expectedReaward((i, j), move)
                self.expectedReturn[-1][j][i] += self.domain.getGamma() * (1 - self.domain.getBeta()) * previousExpectedReturn[newPosition[1]][newPosition[0]]
                self.expectedReturn[-1][j][i] += self.domain.getGamma() * self.domain.getBeta() * previousExpectedReturn[0][0]
        return self.expectedReturn[-1]

    def computeExpectedReturnState(self, position, n):
        return self.computeExpectedReturnMatrix(n)[position[1]][position[0]]
