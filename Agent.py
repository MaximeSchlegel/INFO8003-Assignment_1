import numpy as np
import random as rdm
from MDP_Domain import MDP_Domain


class Agent:

    def __init__(self, domain, policy, useMDPEmulation=False, iterExplore=1000):
        self.domain = domain
        self.emulatedDomain = domain
        self.policy = policy
        self.initialPolicy = policy
        x_max, y_max = self.domain.getShape()
        self.expectedReturn = [np.array([[0. for i in range(x_max)] for j in range(y_max)])]
        self.actionSate = [np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for i in range(x_max)] for j in range(y_max)])]
        if useMDPEmulation:
            self.emulatedDomain = MDP_Domain()
            self.explore(iterExplore)
    def getDomain(self):
        return self.domain

    def getEmulatedDomain(self):
        assert type(self.emulatedDomain) == MDP_Domain, "MDP Domain emulation is nit activated"
        return self.emulatedDomain

    def getPolicy(self):
        return self.policy

    def getExpectedReturn(self):
        return self.expectedReturn

    def getActionState(self):
        return self.actionSate

    def play(self, nbMoves, position=(0, 0), display=False):
        position = position  # the origin is on the top left of the board
        ht = [position]
        cumulatedReward = 0
        for i in range(nbMoves):
            move = self.policy(self.domain, position)
            ht.append(move)
            reward = self.domain.reward(position, move)
            ht.append(reward)
            cumulatedReward += reward
            position = self.domain.move(position, move)
            ht.append(position)
            self.domain.drawNoise()
            if display:
                print("Turn " + str(i))
                print("   Use move : " + str(move))
                print("   Get reward: " + str(reward))
                print("   End the turn in positon : " + str(position))
        if display:
            print("Cumulated Reward : " + str(cumulatedReward))
            print("End Position : " + str(position))
        return ht

    def computeExpectedReturn(self, n, state=None):
        if len(self.expectedReturn) > n:
            if state is not None:
                return self.expectedReturn[n][state[0]][state[1]]
            return self.expectedReturn[n]
        x_max, y_max = self.domain.getShape()
        for it in range(n+1-len(self.expectedReturn)):
            previousExpectedReturn = self.expectedReturn[it-1]
            self.expectedReturn.append(np.array([[0. for i in range(x_max)] for j in range(y_max)]))
            for i in range(x_max):
                for j in range(y_max):
                    move = self.policy(self.domain, (i, j))
                    self.expectedReturn[-1][j][i] = self.emulatedDomain.expectedReward((i, j), move)
                    finalPositons = self.emulatedDomain.moveResult((i, j), move)
                    for newPosition, probability in finalPositons:
                        self.expectedReturn[-1][j][i] += self.domain.getGamma() * probability * previousExpectedReturn[newPosition[1]][newPosition[0]]
        if state is not None:
            return self.expectedReturn[-1][state[0]][state[1]]
        return self.expectedReturn[-1]

    def computeErrorExpectedReturn(self, n):
        return (pow(self.domain.getGamma(), n) * self.emulatedDomain.getMaxReward()) / (1 - self.domain.getGamma())

    def approximateJ(self, error=0.0001):
        n = 0
        currentError = self.computeErrorExpectedReturn(n)
        while currentError >= error:
            n += 1
            self.computeExpectedReturn(n)
            currentError = self.computeErrorExpectedReturn(n)
        return self.computeExpectedReturn(n), currentError

    def displayExpectedRetrun(self, n=None):
        print("Matrix of the Expected Return :")
        if n is None:
            a = 0
            b = len(self.expectedReturn)
        elif n == -1:
            a = len(self.expectedReturn) - 1
            b = len(self.expectedReturn)
        else:
            a = n
            b = n + 1
        for i in range(a, b):
            print("N = " + str(i))
            print("Error = " + str(self.computeErrorExpectedReturn(i)))
            with np.printoptions(precision=2):
                print(self.expectedReturn[i], "\n")

    def computeActionState(self, n, state=None):
        if len(self.actionSate) > n:
            if state is not None:
                return self.actionSate[n][state[0]][state[1]]
            return self.actionSate[n]
        x_max,y_max = self.domain.getShape()
        for it in range(n+1-len(self.actionSate)):
            previousActionState = self.actionSate[it-1]
            self.actionSate.append(np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for i in range(x_max)] for j in range(y_max)]))
            for i in range(x_max):
                for j in range(y_max):
                    for move in self.domain.VALID_ACTIONS:
                        self.actionSate[-1][j][i][move] += self.emulatedDomain.expectedReward((i, j), move)
                        finalPositon = self.emulatedDomain.moveResult((i, j), move)
                        for newPosition, probability in finalPositon:
                            self.actionSate[-1][j][i][move] +=  self.domain.getGamma() * probability * max(previousActionState[newPosition[1]][newPosition[0]].values())
        if state is not None:
            return self.actionSate[-1][state[0]][state[1]]
        return self.actionSate[-1]

    def computeErrorActionState(self, n):
        return (2 * pow(self.domain.getGamma(), n) * self.emulatedDomain.getMaxReward()) / (pow(1 - self.domain.getGamma(), 2))

    def approximatQ(self, error=0.0001):
        n = 0
        currentError = self.computeErrorActionState(n)
        while currentError >= error:
            n += 1
            self.computeActionState(n)
            currentError = self.computeErrorActionState(n)
        return self.computeActionState(n), currentError

    def displayActionState(self, n=None):
        print("Matrix of the Action-State :")
        if n is None:
            a = 0
            b = len(self.actionSate)
        elif n == -1:
            a = len(self.actionSate) - 1
            b = len(self.actionSate)
        else:
            assert n > 0
            assert n < len(self.actionSate)
            a = n
            b = n + 1
        for i in range(a, b):
            print("N = " + str(i))
            print("Error = " + str(self.computeErrorActionState(i)))
            with np.printoptions(precision=2):
                print(self.actionSate[i], "\n")

    def extractOptimalPolicy(self, error = 0.0001, use=False):
        n = len(self.actionSate)
        if self.computeErrorActionState(n) > error:
            self.approximatQ(error)
        x_max, y_max = self.domain.getShape()
        finalQ = self.actionSate[-1]
        optimalPolicy = [[max(finalQ[j][i], key=finalQ[j][i].get) for i in range(x_max)] for j in range(y_max)]
        def newPolicy(domain, position):
            arg = optimalPolicy
            return arg[position[1]][position[0]]
        if use:
            self.policy = newPolicy
        return newPolicy

    def displayPolicy(self):
        print('Current Policy :')
        x_max, y_max = self.domain.getShape()
        visualisation = [[self.policy(self.domain, (i, j)) for i in range(x_max)] for j in range(y_max)]
        for j in range(y_max):
            print(visualisation[j])
        print()

    def restoreInitialPolicy(self):
        self.policy = self.initialPolicy

    def explore(self, n=1000):
        assert type(self.emulatedDomain) == MDP_Domain, "MDP Domain Emlation is not activated"
        seen = set()
        def explorePolicy(domain, positon):
            arg = seen
            for action in domain.VALID_ACTIONS:
                if not((positon, action) in seen):
                    arg.add((positon, action))
                    return action
            return rdm.choice(domain.VALID_ACTIONS)
        oldpolicy = self.policy
        self.policy = explorePolicy
        x_max, y_max = self.domain.getShape()
        ht = self.play(n, (rdm.randint(0,x_max), rdm.randint(0,y_max)))
        self.emulatedDomain.analyzeArray(ht)
        self.policy = oldpolicy
