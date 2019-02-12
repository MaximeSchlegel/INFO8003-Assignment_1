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

myDomain = Domain(myBoard, .25)
#Here the domain is Stochastic to get a Determistic Domain beta must be 0
myGame = Game(myDomain, upPolicy)

myGame.play(10)

myGame.computeExpectedReturnMatrix(5)
myGame.displayExpectedRetrun(3)

myGame.approximateJ(0.1)
myGame.displayExpectedRetrun()

