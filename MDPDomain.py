class MDPDomain:

    def __init__(self, ht=None):
        self.maxReward = 0
        self.visited = {}
        # key: initialPostion, move || value: number of times this state has been visited
        self.reward = {}
        # key: initialPostion, move || value: cumulated reward received the move from initialPosition was taken
        self.probability = {}
        # key: initialPostion, move || value: dictionnary with key: finalPosition, and value number of times this has happened
        if ht is not None:
            self.analyzeArray(ht)

    def getMaxReward(self):
        return self.maxReward

    def analyze(self, initialPosition, move, reward, finalPosition):
        id = initialPosition, move
        if reward > self.maxReward:
            self.maxReward = reward
        if not (id in self.visited.keys()):
            self.visited[id] = 0
            self.reward[id] = 0
            self.probability[id] = {}
            self.probability[id][finalPosition] = 0
        elif not(finalPosition in self.probability[id].keys()):
            self.probability[id][finalPosition] = 0
        self.visited[id] += 1
        self.reward[id] += reward
        self.probability[id][finalPosition] += 1

    def analyzeArray(self, ht):
        assert (len(ht) - 1) % 3 == 0
        for i in range((len(ht) - 1) // 3):
            initialPosition, move, reward, finalPosition = ht[3 * i], ht[3 * i + 1], ht[3 * i + 2], ht[3 * i + 3]
            self.analyze(initialPosition, move, reward, finalPosition)

    def expectedReward(self, position, move):
        id = position, move
        if not(id in self.reward) or not(id in self.visited):
            return 0
        return self.reward[id] / self.visited[id]

    def computeProbability(self, initialPosition, move, finalPosition):
        id = initialPosition, move
        if not(id in self.probability) or not(finalPosition in self.probability[id]) or not(id in self.visited):
            return 0
        return self.probability[id][finalPosition] / self.visited[id]

    def moveResult(self, initialPosition, move):
        res = []
        if not (initialPosition, move) in self.probability:
            return []
        finalPositions = self.probability[initialPosition, move].keys()
        for position in finalPositions:
            res.append((position, self.computeProbability(initialPosition, move, position)))
        return res
