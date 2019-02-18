import numpy as np
from Domain import Domain
from Agent import Agent


testBoard = np.array([[i+j for i in range(5)] for j in range(5)])

def downPolicy(domain: Domain, position):
    return domain.DOWN


def test_creationAgent():
    testDomain = Domain(testBoard)
    testAgent = Agent(testDomain, downPolicy)
    assert testAgent is not None
    assert testAgent.getDomain() == testDomain
    assert testAgent.getPolicy() == downPolicy
    tmp = testAgent.getExpectedReturn()
    assert tmp[0][0][0] == 0


def test_play():
    testDomain = Domain(testBoard)
    testAgent = Agent(testDomain, downPolicy)
    ht = testAgent.play(1)
    assert ht == [(0, 0), (0, 1), 1, (0, 1)]
    ht = testAgent.play(5)
    assert ht == [(0, 0), (0, 1), 1, (0, 1), (0, 1), 2, (0, 2), (0, 1), 3, (0, 3), (0, 1), 4, (0, 4), (0, 1), 4, (0, 4)]


def test_computeExpectedReturn():
    testDomain = Domain(testBoard, 0.25)
    testAgent = Agent(testDomain, downPolicy)
    testAgent.computeExpectedReturn(1)
    tmp = testAgent.getExpectedReturn()
    assert len(tmp) == 2
    assert tmp[-1][0][0] == 0.75
    assert tmp[-1][4][4] == 6
    tmp = testAgent.getExpectedReturn()
    testAgent.computeExpectedReturn(3)
    assert len(tmp) == 4


def test_computeErrorExpectedReturn():
    testDomain = Domain(testBoard)
    testAgent = Agent(testDomain, downPolicy)
    assert testAgent.computeErrorExpectedReturn(0) - 800 < 0.001
    assert testAgent.computeErrorExpectedReturn(5) - 760.792 < 0.001


def test_approximateJ():
    testDomain = Domain(testBoard)
    testAgent = Agent(testDomain, downPolicy)
    testAgent.approximateJ(1)
    assert len(testAgent.getExpectedReturn()) == 667
