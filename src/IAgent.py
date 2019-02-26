import numpy as np
import random as rdm
from src.QAgent import QAgent


class IAgent(QAgent):

    def __init__(self, domain, epsilon, alpha=0.05, replay=False, max_sample=None):
        super().__init__(domain, self.get_move, alpha, replay, max_sample )
        self.epsilon = epsilon

    def random_policy(self, domain, position):
        return rdm.choice(domain.VALID_ACTIONS)

    def get_move(self, domain, position):
        ex = rdm.random()
        if ex < self.epsilon:
            return self.random_policy(domain, position)
        return self.policy_mu(domain, position)

    def explore_and_analyze(self, n=1000):
        x_max, y_max = self.domain.get_shape()
        self.train(1, n, (rdm.randint(0, x_max - 1), rdm.randint(0, y_max - 1)))

    def train(self, nb_batch, batch_size, start_position=None, display=False):
        for i in range(nb_batch):
            ht = self.play(batch_size, position=start_position)
            self.analyze_trajectory(ht, self.experience_replay, self.max_sample)
            if display:
                self.action_state_display()
                self.policy_display()
