from MDPDomain import MDPDomain

ht = [(3, 2), (0, -1), 9, (3, 1), (0, -1), -3, (0, 0), (0, -1), -3, (0, 0), (1, 0), 1, (1, 0), (0, -1), 1, (1, 0),
      (1, 0), -5, (2, 0), (0, -1), -5, (2, 0), (1, 0), -3, (0, 0), (0, 1), 6, (0, 1), (0, -1), -3, (0, 0), (-1, 0), -3,
      (0, 0), (1, 0), 1, (1, 0), (0, 1), 3, (1, 1), (0, -1), -3, (0, 0), (-1, 0), -3, (0, 0), (0, 1), 6, (0, 1), (1, 0),
      -3, (0, 0), (0, 1), 6, (0, 1), (0, 1), 5, (0, 2), (0, -1), 6, (0, 1), (-1, 0), 6, (0, 1), (0, 1), 5, (0, 2),
      (1, 0), -3, (0, 0), (1, 0), 1, (1, 0), (-1, 0), -3, (0, 0), (0, 1), 6, (0, 1), (1, 0), 3, (1, 1)]

def test_creationMDP():
      testMDP = MDPDomain()
      assert testMDP.maxReward == 0
      assert testMDP.visited == {}
      assert testMDP.reward == {}
      assert testMDP.probability == {}


def test_getMAxReward():
      testMDP = MDPDomain(ht)
      assert testMDP.getMaxReward() == testMDP.maxReward


def test_analyze():
      testMDP = MDPDomain()
      testMDP.analyze((0, 0), (0, 1), 5, (0, 1))
      assert testMDP.visited[((0, 0), (0, 1))] == 1
      assert testMDP.reward[((0, 0), (0, 1))] == 5
      assert testMDP.probability[((0, 0), (0, 1))] == {(0, 1): 1}


def test_analyzeArray():
      testMDP = MDPDomain()
      testMDP.analyzeArray(ht)
      assert testMDP.visited != {}
      assert testMDP.probability != {}
      assert testMDP.reward != {}
      assert testMDP.getMaxReward() == max([i if type(i) == int else 0 for i in ht])

