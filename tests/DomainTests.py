import numpy as np
from src.Domain import Domain

testBoard = np.array([[i+j for i in range(5)] for j in range(5)])


def test_createDomain():
    testDomain = Domain(testBoard, .25, .5)
    assert testDomain is not None
    assert np.array_equal(testDomain.board, testBoard)
    assert testDomain.get_shape() == (5, 5)
    assert testDomain.beta == 0.25
    assert testDomain.get_gamma() == 0.5


def test_deterministicMove_normalMove():
    testDomain = Domain(testBoard, .25)
    move = (1, 0)
    pos = testDomain.deterministic_move((0, 0), move)
    assert pos == move


def test_deterministicMove_againstWall():
    testDomain = Domain(testBoard)
    pos = testDomain.deterministic_move((0, 0), testDomain.UP)
    assert pos == (0, 0)
    pos = testDomain.deterministic_move((0, 0), testDomain.LEFT)
    assert pos == (0, 0)
    pos = testDomain.deterministic_move((4, 4), testDomain.RIGHT)
    assert pos == (4, 4)
    pos = testDomain.deterministic_move((4, 4), testDomain.DOWN)
    assert pos == (4, 4)


def test_deterministicReward():
    testDomain = Domain(testBoard)
    assert testDomain.deterministic_reward((0, 0), testDomain.RIGHT) == 1
    assert testDomain.deterministic_reward((0, 0), testDomain.LEFT) == 0


def test_stochasticMove():
    testDomain = Domain(testBoard, .5)
    testDomain.noise = 0.5
    pos = testDomain.move((0, 0), testDomain.RIGHT)
    assert pos == (1, 0)
    testDomain.noise = 0.9
    pos = testDomain.move((0, 0), testDomain.RIGHT)
    assert pos == (0, 0)


def test_reward():
    testDomain = Domain(testBoard, .5)
    testDomain.noise = 0.5
    reward = testDomain.reward((0, 0), testDomain.RIGHT)
    assert reward == 1
    testDomain.noise = 0.9
    reward = testDomain.reward((0, 0), testDomain.RIGHT)
    assert reward == 0

