import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from Domain import Domain
from MDPDomain import MDPDomain
from Agent import Agent


myBoard = np.array([[-3,   1,  -5,  0,   19],
                    [ 6,   3,   8,  9,   10],
                    [ 5,  -8,   4,  1,  -8],
                    [ 6,  -9,   4,  19, -5],
                    [-20, -17, -4, -3,   9]])
print(myBoard)

def upPolicy(domain, position):
    return domain.UP


def randomPolicy(domain, state):
    return rdm.choice(Domain.VALID_ACTIONS)


seen = set()
def explorePolicy(domain, positon):
    arg = seen
    for action in domain.VALID_ACTIONS:
        if not ((positon, action) in seen):
            arg.add((positon, action))
            return action
    return rdm.choice(domain.VALID_ACTIONS)


# Deterministic Domain without MDP Domain Emulation
print("Deterministic Domain without MDP Domain Emulation : ", '\n')
DDomain = Domain(myBoard)
DAgent = Agent(DDomain, upPolicy)

# DAgent.play(10, display=True)

DAgent.computeExpectedReturn(1)
DAgent.displayExpectedRetrun(-1)
DAgent.approximateJ()
DAgent.displayExpectedRetrun(-1)

DAgent.computeActionState(1)
DAgent.displayActionState(-1)
DAgent.approximateQ()
DAgent.displayActionState(-1)

DAgent.extractOptimalPolicy(use=True)
DAgent.displayPolicy()

DAgent.approximateJ(erase=True)
DAgent.displayExpectedRetrun(-1)

# DAgent.play(10, display=True)


# Stochastic Domain without MDP Domain Emulation
print("\n\nStochastic Domain without MDP Domain Emulation : ", '\n')
SDomain = Domain(myBoard, .25)
SAgent = Agent(SDomain, upPolicy)

# SAgent.play(10, display=True)

SAgent.computeExpectedReturn(1)
SAgent.displayExpectedRetrun(-1)
SAgent.approximateJ()
SAgent.displayExpectedRetrun(-1)

SAgent.computeActionState(1)
SAgent.displayActionState(-1)
SAgent.approximateQ()
SAgent.displayActionState(-1)

SAgent.extractOptimalPolicy(use=True)
DAgent.displayPolicy()

SAgent.approximateJ(erase=True)
SAgent.displayExpectedRetrun(-1)

# DAgent.play(10, display=True)


# Derterminist Domain with MDP Domain Emulation
print("\n\nDeterministic Domain with MDP Domain Emulation : ", '\n')
DMDPAgent = Agent(DDomain, upPolicy, useMDPEmulation=True)

# DMDPAgent.play(10, display=True)

DMDPAgent.computeExpectedReturn(1)
DMDPAgent.displayExpectedRetrun(-1)
DMDPAgent.approximateJ()
DMDPAgent.displayExpectedRetrun(-1)

DMDPAgent.computeActionState(1)
DMDPAgent.displayActionState(-1)
DMDPAgent.approximateQ()
DMDPAgent.displayActionState(-1)

DMDPAgent.extractOptimalPolicy(use=True)
DMDPAgent.displayPolicy()

DMDPAgent.approximateJ(erase=True)
DMDPAgent.displayExpectedRetrun(-1)

# DMDPAgent.play(10, display=True)


# Stochastic Domain with MDP Domain Emulation
print("\n\nStochastic Domain with MDP Domain Emulation : ", '\n')
SMDPAgent = Agent(SDomain, upPolicy, useMDPEmulation=True)

# SMDPAgent.play(10, display=True)

SMDPAgent.computeExpectedReturn(1)
SMDPAgent.displayExpectedRetrun(-1)
SMDPAgent.approximateJ()
SMDPAgent.displayExpectedRetrun(-1)

SMDPAgent.computeActionState(1)
SMDPAgent.displayActionState(-1)
SMDPAgent.approximateQ()
SMDPAgent.displayActionState(-1)

SMDPAgent.extractOptimalPolicy(use=True)
SMDPAgent.displayPolicy()

SMDPAgent.approximateJ(erase=True)
SMDPAgent.displayExpectedRetrun(-1)

# SMDPAgent.play(10, display=True)


# rdmAgent = Agent(SDomain,randomPolicy)
# exploAgent = Agent(SDomain, explorePolicy)
#
# def convSpeed(Agent):
#     hts = Agent.play(10000)
#     htl = []
#     rSpeed = []
#     pSpeed = []
#     for k in range(100):
#         seen = set()
#         ht = hts[:(k * 300) + 31]
#         htl.append(len(ht))
#         testDomain = MDPDomain(ht)
#         proba = 0
#         reward = 0
#         for i in range(5):
#             for j in range(5):
#                 for action in Domain.VALID_ACTIONS:
#                     reward += SDomain.expectedReward((i, j), action) - testDomain.expectedReward((i, j), action)
#                     final1 = SDomain.moveResult((i, j), action)
#                     final2 = testDomain.moveResult((i, j), action)
#                     for pos, prob in final2:
#                         for l in range(len(final1)):
#                             if final1[l][0] == pos:
#                                 proba += final1[l][1] - prob
#                                 final1.pop(l)
#                                 break;
#                     for pos,prob in final1:
#                         proba += prob
#         rSpeed.append(reward)
#         pSpeed.append(proba)
#     return htl, rSpeed, pSpeed
# htl, rdmRSpeed, rdmPSpeed = convSpeed(rdmAgent)
# htl, exploRSpeed, exploPSpeed = convSpeed(exploAgent)
# plt.plot(htl,rdmRSpeed, 'r--', label="Reward for the Rdm Player")
# plt.plot(htl, rdmPSpeed, 'r', label="Probability for the Rdm Player")
# plt.plot(htl, exploRSpeed, 'b--', label="Reward Convergence for the Explo Player")
# plt.plot(htl, exploPSpeed, 'b', label="Probability Convergence for the Explo Player")
# plt.xlabel("Length of the History")
# plt.ylabel("Cumulated Error")
# plt.grid()
# plt.legend()
# plt.show()
