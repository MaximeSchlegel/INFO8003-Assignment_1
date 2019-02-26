import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from src.Domain import Domain
from src.MDPDomain import MDPDomain
from src.Agent import Agent
from src.MDPAgent import MDPAgent
from src.QAgent import QAgent
from src.IAgent import IAgent


myBoard = np.array([[-3,   1,  -5,  0,   19],
                    [ 6,   3,   8,  9,   10],
                    [ 5,  -8,   4,  1,  -8],
                    [ 6,  -9,   4,  19, -5],
                    [-20, -17, -4, -3,   9]])
print(myBoard)


def randomPolicy(domain, state):
    return rdm.choice(Domain.VALID_ACTIONS)


def explore_policy():
    seen = set()

    def explore(domain, position):
        not_take = []
        for action in domain.VALID_ACTIONS:
            if not ((position, action) in seen):
                not_take.append(action)
        if not_take:
            action = rdm.choice(not_take)
            seen.add((position, action))
            return action
        return rdm.choice(domain.VALID_ACTIONS)

    return explore


DDomain = Domain(myBoard)
SDomain = Domain(myBoard, .25)

def conv_speed_rp(agent):
    hts = agent.play(10000)
    htl = []
    r_speed = []
    p_speed = []
    for k in range(100):
        seen = set()
        ht = hts[:(k * 300) + 31]
        htl.append(len(ht))
        test_domain = MDPDomain(ht)
        probability = 0
        reward = 0
        for i in range(5):
            for j in range(5):
                for action in Domain.VALID_ACTIONS:
                    reward += SDomain.expected_reward((i, j), action) - test_domain.expected_reward((i, j), action)
                    final1 = SDomain.expected_move((i, j), action)
                    final2 = test_domain.expected_move((i, j), action)
                    for pos, proba in final2:
                        for l in range(len(final1)):
                            if final1[l][0] == pos:
                                probability += final1[l][1] - proba
                                final1.pop(l)
                                break
                    for pos, proba in final1:
                        probability += proba
        r_speed.append(reward)
        p_speed.append(probability)
    return htl, r_speed, p_speed


def conv_speed_q(agent):
    x_max, y_max = SDomain.get_shape()
    ref_agent = Agent(SDomain, randomPolicy)
    ref_agent.action_state_approximate()
    ht1 = []
    q_speed = []
    for it in range(100):
        error = 0
        agent.explore_and_analyze(1000)
        agent.action_state_approximate(erase=True)
        q = agent.get_action_state_matrix(-1)
        qr = ref_agent.get_action_state_matrix(-1)
        for i in range(x_max):
            for j in range(y_max):
                for k in SDomain.VALID_ACTIONS:
                    error += qr[j][i][k] - q[j][i][k]
        ht1.append(it * 100)
        q_speed.append(error)
    return ht1, q_speed


# rdmAgent = Agent(SDomain,randomPolicy)
# exploAgent = Agent(SDomain, explore_policy())
#
# htl, rdmRSpeed, rdmPSpeed = conv_speed_rp(rdmAgent)
# htl, exploRSpeed, exploPSpeed = conv_speed_rp(exploAgent)
# plt.plot(htl,rdmRSpeed, 'r--', label="Reward for the Rdm Player")
# plt.plot(htl, rdmPSpeed, 'r', label="Probability for the Rdm Player")
# plt.plot(htl, exploRSpeed, 'b--', label="Reward Convergence for the Explo Player")
# plt.plot(htl, exploPSpeed, 'b', label="Probability Convergence for the Explo Player")
# plt.xlabel("Length of the History")
# plt.ylabel("Cumulated Error")
# plt.grid()
# plt.legend()
# plt.show()

MDPAgent = MDPAgent(SDomain, randomPolicy)
QLAgent = QAgent(SDomain, randomPolicy)
QLRAgent = QAgent(SDomain, randomPolicy, replay=True, max_sample=100)

plt.figure(2)

hti, ierror = conv_speed_q(QLRAgent)
htm, mdperror = conv_speed_q(MDPAgent)
htq, qerror = conv_speed_q(QLAgent)

IAgent.action_state_display()


plt.plot(htm, mdperror, 'r', label="MDP Agent")
plt.plot(htq, qerror, 'b', label="Q-learning Agent")
plt.plot(hti, ierror, 'g', label="Q-learning Agent with experience replay (100 samples)")
plt.xlabel("Length of the History")
plt.ylabel("Cumulated Error On Q")
plt.grid()
plt.legend()
plt.show()
