import numpy as np
import random as rdm
from src.Domain import Domain
from src.Agent import Agent
from src.MDPAgent import MDPAgent
from src.QAgent import QAgent
from src.IAgent import IAgent


myBoard = np.array([[-3,   1,  -5,  0,   19],
                    [ 6,   3,   8,  9,   10],
                    [ 5,  -8,   4,  1,  -8],
                    [ 6,  -9,   4,  19, -5],
                    [-20, -17, -4, -3,   9]])


def up_policy(domain, position):
    return domain.UP


def random_policy(domain, state):
    return rdm.choice(Domain.VALID_ACTIONS)


DDomain = Domain(myBoard)
SDomain = Domain(myBoard, .25)


print("Board :")
print(myBoard)

# Deterministic Domain without MDP Domain Emulation
print("\n\nDeterministic Domain without MDP Domain Emulation :", '\n')
DAgent = Agent(DDomain, up_policy)

# DAgent.play(10, display=True)

DAgent.expected_return_compute(1)
DAgent.expected_return_display(-1)
DAgent.expected_return_approximate()
DAgent.expected_return_display(-1)

DAgent.action_state_compute(1)
DAgent.action_state_display(-1)
DAgent.action_state_approximate()
DAgent.action_state_display(-1)

DAgent.policy_get_mu(use=True)
DAgent.policy_display()

DAgent.expected_return_approximate(erase=True)
DAgent.expected_return_display(-1)

# DAgent.play(10, display=True)


# Stochastic Domain without MDP Domain Emulation
print("\n\nStochastic Domain without MDP Domain Emulation :", '\n')
SAgent = Agent(SDomain, up_policy)

# SAgent.play(10, display=True)

SAgent.expected_return_compute(1)
SAgent.expected_return_display(-1)
SAgent.expected_return_approximate()
SAgent.expected_return_display(-1)

SAgent.action_state_compute(1)
SAgent.action_state_display(-1)
SAgent.action_state_approximate()
SAgent.action_state_display(-1)

SAgent.policy_get_mu(use=True)
DAgent.policy_display()

SAgent.expected_return_approximate(erase=True)
SAgent.expected_return_display(-1)

# DAgent.play(10, display=True)


# Derterminist Domain with MDP Domain Emulation
print("\n\nDeterministic Domain with MDP Domain Emulation :", '\n')
DMDPAgent = MDPAgent(DDomain, up_policy)

# DMDPAgent.play(10, display=True)

DMDPAgent.expected_return_compute(1)
DMDPAgent.expected_return_display(-1)
DMDPAgent.expected_return_approximate()
DMDPAgent.expected_return_display(-1)

DMDPAgent.action_state_compute(1)
DMDPAgent.action_state_display(-1)
DMDPAgent.action_state_approximate()
DMDPAgent.action_state_display(-1)

DMDPAgent.policy_get_mu(use=True)
DMDPAgent.policy_display()

DMDPAgent.expected_return_approximate(erase=True)
DMDPAgent.expected_return_display(-1)

# DMDPAgent.play(10, display=True)


# Stochastic Domain with MDP Domain Emulation
print("\n\nStochastic Domain with MDP Domain Emulation :", '\n')
SMDPAgent = MDPAgent(SDomain, up_policy)

# SMDPAgent.play(10, display=True)

SMDPAgent.expected_return_compute(1)
SMDPAgent.expected_return_display(-1)
SMDPAgent.expected_return_approximate()
SMDPAgent.expected_return_display(-1)

SMDPAgent.action_state_compute(1)
SMDPAgent.action_state_display(-1)
SMDPAgent.action_state_approximate()
SMDPAgent.action_state_display(-1)

SMDPAgent.policy_get_mu(use=True)
SMDPAgent.policy_display()

SMDPAgent.expected_return_approximate(erase=True)
SMDPAgent.expected_return_display(-1)

# SMDPAgent.play(10, display=True)


# Deterministic domain with Q learning
print("\n\nDeterministic domain with Q learning :", '\n')
DQAgent = QAgent(DDomain, random_policy)

DQAgent.analyze_trajectory(DQAgent.play(1000))

DQAgent.action_state_display()

DQAgent.expected_return_approximate()
DQAgent.expected_return_display(-1)

DQAgent.policy_get_mu(use=True)
DQAgent.policy_display()

DQAgent.expected_return_approximate(erase=True)
DQAgent.expected_return_display(-1)


# Stochastic Domain with Q-learning
print("\n\nStochastic domain with Q learning :", '\n')
SQAgent = QAgent(SDomain, random_policy)

SQAgent.analyze_trajectory(SQAgent.play(1000))

SQAgent.action_state_display()

SQAgent.expected_return_approximate()
SQAgent.expected_return_display(-1)

SQAgent.policy_get_mu(use=True)
SQAgent.policy_display()

SQAgent.expected_return_approximate(erase=True)
SQAgent.expected_return_display(-1)

# Intelligent Agent on a deterministic domain

IAgent = IAgent(DDomain, )