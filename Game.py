import numpy as np


class Game:

    def __init__(self, domain, policy):
        self.domain = domain
        self.policy = policy
        self.expectedReturn = [np.array([[0. for i in range(self.domain.x_max)] for j in range(self.domain.y_max)])]

    def getDomain(self):
        return self.domain

    def getPolicy(self):
        return self.policy

    def getExpectedReturn(self):
        return self.expectedReturn

    def play(self, nbMoves, position=(0, 0), display=False):
        position = position # the origin is on the top left of the board
        cumulatedReward = 0
        for i in range(nbMoves):
            move = self.policy(self.domain, position)
            cumulatedReward += self.domain.reward(position, move)
            position = self.domain.move(position, move)
            self.domain.drawNoise()
            if display:
                print("Turn " + str(i))
                print("   Use move : " + str(move))
                print("   End the turn in positon : " + str(position))
        if display:
            print("Cumulated Reward : " + str(cumulatedReward))
            print("End Position : " + str(position))
        return position, cumulatedReward

    def computeError(self, n):
        Br = np.max(self.domain.getBoard())
        return (pow(self.domain.getGamma(), n) * Br) / (1 - self.domain.getGamma())

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

    def approximateJ(self, error):
        n = 0
        currentError = self.computeError(n)
        while currentError >= error:
            n += 1
            self.computeExpectedReturnMatrix(n)
            currentError = self.computeError(n)
        return self.computeExpectedReturnMatrix(n), currentError

    def displayExpectedRetrun(self, n=None):
        print("Matrix of the Expected Return :")
        if n is None:
            for i in range(len(self.expectedReturn)):
                print("N = " + str(i))
                print("Error = " + str(self.computeError(i)))
                print(self.expectedReturn[i], "\n")
        else:
            if n == -1:
                n = len(self.expectedReturn) - 1
            print("N = " + str(n))
            print("Error = " + str(self.computeError(n)))
            print(self.expectedReturn[n])