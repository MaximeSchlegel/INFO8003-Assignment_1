import numpy as np
from Domain import Domain
from Game import Game


testBoard = np.array([[i+j for i in range(5)] for j in range(5)])

def downPolicy(domain: Domain, position):
    return domain.DOWN


def test_creationGame():
    testDomain = Domain(testBoard)
    testGame = Game(testDomain, downPolicy)
    print("\n")
    print(testBoard)
    assert testGame is not None
    assert testGame.domain == testDomain
    assert testGame.policy == downPolicy
    tmp = testGame.getExpectedReturn()
    assert tmp[0][0][0] == 0


def test_play():
    testDomain = Domain(testBoard)
    testGame = Game(testDomain, downPolicy)
    position, cumulatedReward = testGame.play(1)
    assert position == (0, 1)
    assert cumulatedReward == 1
    position, cumulatedReward = testGame.play(5)
    assert position == (0, 4)
    assert cumulatedReward == 14


def test_computeExpectedReturn():
    testDomain = Domain(testBoard, 0.25)
    testGame = Game(testDomain, downPolicy)
    testGame.computeExpectedReturnMatrix(1)
    tmp = testGame.getExpectedReturn()
    assert len(tmp) == 2
    assert tmp[-1][0][0] == 0.75
    assert tmp[-1][4][4] == 6
    testGame.computeExpectedReturnMatrix(3)
    tmp  = testGame.getExpectedReturn()
    assert len(tmp) == 4
