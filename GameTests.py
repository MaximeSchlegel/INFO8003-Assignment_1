import numpy as np
from Domain import Domain
from Game import Game


testBoard = np.array([[i+j for i in range(5)] for j in range(5)])

def downPolicy(domain: Domain, position):
    return domain.DOWN


def test_creationGame():
    testDomain = Domain(testBoard)
    testGame = Game(testDomain, downPolicy)
    assert testGame is not None
    assert testGame.getDomain() == testDomain
    assert testGame.getPolicy() == downPolicy
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
    tmp = testGame.getExpectedReturn()
    assert len(tmp) == 4


def test_computeError():
    testDomain = Domain(testBoard)
    testGame = Game(testDomain, downPolicy)
    assert testGame.computeError(0) - 800 < 0.001
    assert testGame.computeError(5) - 760.792 < 0.001


def test_approximateJ():
    testDomain = Domain(testBoard)
    testGame = Game(testDomain, downPolicy)
    testGame.approximateJ(1)
    assert len(testGame.getExpectedReturn()) == 667
