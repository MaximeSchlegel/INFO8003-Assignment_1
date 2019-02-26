from src.utils import decompose_trajectory

class MDPDomain:

    def __init__(self, trajectory=None):
        self.max_reward = 0
        self.visited = {}
        # key: initialPostion, move || value: number of times this state has been visited
        self.reward = {}
        # key: initialPostion, move || value: cumulated reward received the move from initialPosition was taken
        self.probability = {}
        # key: initialPostion, move || value: dictionnary with key: finalPosition, and value number of times this has happened
        if trajectory is not None:
            self.analyze_trajectory(trajectory)

    def get_max_reward(self):
        return self.max_reward

    def analyze_one_step_system_transition(self, initial_position, move, reward, final_position):
        id = initial_position, move
        if reward > self.max_reward:
            self.max_reward = reward
        if not (id in self.visited.keys()):
            self.visited[id] = 0
            self.reward[id] = 0
            self.probability[id] = {}
            self.probability[id][final_position] = 0
        elif not(final_position in self.probability[id].keys()):
            self.probability[id][final_position] = 0
        self.visited[id] += 1
        self.reward[id] += reward
        self.probability[id][final_position] += 1

    def analyze_trajectory(self, trajectory):
        history = decompose_trajectory(trajectory)
        for i in range(len(history)):
            initial_position, move, reward, final_position = history[i]
            self.analyze_one_step_system_transition(initial_position, move, reward, final_position)

    def compute_move_probability(self, initial_position, move, final_position):
        id = initial_position, move
        if not(id in self.probability) or not(final_position in self.probability[id]) or not(id in self.visited):
            return 0
        return self.probability[id][final_position] / self.visited[id]

    def expected_move(self, initial_position, move):
        res = []
        if not (initial_position, move) in self.probability:
            return []
        finalPositions = self.probability[initial_position, move].keys()
        for position in finalPositions:
            res.append((position, self.compute_move_probability(initial_position, move, position)))
        return res

    def expected_reward(self, position, move):
        id = position, move
        if not(id in self.reward) or not(id in self.visited):
            return 0
        return self.reward[id] / self.visited[id]
