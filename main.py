import numpy as np
from DeterministicDomain import DeterministicDomain


myBoard = np.array([[-3, 1, -5, 0, 19],
                    [6, 3, 8, 9, 10],
                    [5, -8, 4, 1, -8],
                    [6, -9, 4, 19, -5],
                    [-20, -17, -4, -3, 9]])
testBoard = np.array([[i+j for i in range(5)] for j in range(5)])
print(myBoard)
print(testBoard)

deterministicDomain = DeterministicDomain(myBoard)
print(deterministicDomain.reward((0, 1)))
