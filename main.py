import numpy as np
from Domain import Domain
from Game import Game


myBoard = np.array([[-3, 1, -5, 0, 19],
                    [6, 3, 8, 9, 10],
                    [5, -8, 4, 1, -8],
                    [6, -9, 4, 19, -5],
                    [-20, -17, -4, -3, 9]])
print(myBoard)


def upPolicy(domain, position):
    return domain.UP

# Deterministic Domain
print("Deterministic Domain : ")
DDomain = Domain(myBoard)
DGame = Game(DDomain, upPolicy)

DGame.play(10, display=True)

DGame.computeExpectedReturnMatrix(5)
DGame.displayExpectedRetrun(2)

DGame.approximateJ(0.1)
DGame.displayExpectedRetrun(-1)

#Stochastic Domain
print("\nStochastic Domain")
SDomain = Domain(myBoard, .25)

SGame = Game(SDomain, upPolicy)

SGame.play(10, display=True)

SGame.computeExpectedReturnMatrix(5)
SGame.displayExpectedRetrun(2)

SGame.approximateJ(0.001)
SGame.displayExpectedRetrun(-1)