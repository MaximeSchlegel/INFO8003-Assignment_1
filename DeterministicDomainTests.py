import numpy as np
from DeterministicDomain import DeterministicDomain

testBoard = np.array([[i+j for i in range(5)] for j in range(5)])


def test_createDomain():
    testDomain = DeterministicDomain(testBoard)
    assert testDomain is not None
    assert np.array_equal(testDomain.board, testBoard)
    assert testDomain.x_max == 5
    assert testDomain.y_max == 5


def test_move_normalMove():
    testDomain = DeterministicDomain(testBoard)
    move = (3, 3)
    pos = testDomain.move((0, 0), move)
    assert pos == move


def test_move_moveAgainstWall():
    testDomain = DeterministicDomain(testBoard)
    pos = testDomain.move((0, 0), testDomain.UP)
    assert pos == (0, 0)
    pos = testDomain.move((0, 0), testDomain.LEFT)
    assert pos == (0, 0)
    pos = testDomain.move((4, 4), testDomain.RIGHT)
    assert pos == (4, 4)
    pos = testDomain.move((4, 4), testDomain.DOWN)
    assert pos == (4, 4)


def test_reward():
    testDomain = DeterministicDomain(testBoard)
    assert testDomain.reward((0, 0), testDomain.RIGHT) == 1
    assert testDomain.reward((0, 0), testDomain.LEFT) == 0
