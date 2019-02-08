class Game:

    def __init__(self, domain, policy):
        self.domain = domain
        self.policy = policy

        self.expectedReturn = [[[0 for i in range(self.domain.x_max)] for j in range(self.domain.y_max)]]

    def play(self, nbMoves):
        position = 0, 0  # the origin is on the top left of the board
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
        return(position, cumulatedReward)

    def computeExpectedReturn(self, n):
        if len(self.expectedReturn) > n:
            return self.expectedReturn[n]

        previousExpectedReturn = self.computeExpectedReturn(n-1)
        self.expectedReturn.append([[0 for i in range(self.domain.x_max)] for j in range(self.domain.y_max)])

        for i in range(self.domain.x_max):
            for j in range(self.domain.y_max):
                move = self.policy(self.domain, (i, j))
                newPosition = self.domain.move((i, j), move)
                self.expectedReturn[-1][j][i] = self.domain.expectedReaward((i, j), move)
                self.expectedReturn[-1][j][i] += self.domain.gamma * (1 - self.domain.beta) * previousExpectedReturn[newPosition[1]][newPosition[0]]
                self.expectedReturn[-1][j][i] += self.domain.gamma * self.domain.beta * previousExpectedReturn[0][0]
        return self.expectedReturn[-1]
