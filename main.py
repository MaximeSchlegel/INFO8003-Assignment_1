import numpy as np
from Domain import Domain
from Agent import Agent


myBoard = np.array([[-3,   1,  -5,  0,   19],
                    [ 6,   3,   8,  9,   10],
                    [ 5,  -8,   4,  1,  -8],
                    [ 6,  -9,   4,  19, -5],
                    [-20, -17, -4, -3,   9]])
print(myBoard)


def upPolicy(domain, position):
    return domain.UP


# Deterministic Domain without MDP Domain Emulation
print("Deterministic Domain without MDP Domain Emulation : ", '\n')
DDomain = Domain(myBoard)
DAgent = Agent(DDomain, upPolicy)

DAgent.play(10, display=True)

DAgent.computeExpectedReturn(1)
DAgent.displayExpectedRetrun(-1)

DAgent.approximateJ()
DAgent.displayExpectedRetrun(-1)

DAgent.computeActionState(1)
DAgent.displayActionState(-1)

DAgent.approximatQ()
DAgent.displayActionState(-1)
DAgent.extractOptimalPolicy(use=True)
DAgent.displayPolicy()

DAgent.play(10, display=True)


# Stochastic Domain without MDP Domain Emulation
print("\n\nStochastic Domain without MDP Domain Emulation : ", '\n')
SDomain = Domain(myBoard, .25)
SAgent = Agent(SDomain, upPolicy)

SAgent.play(10, display=True)

SAgent.computeExpectedReturn(1)
SAgent.displayExpectedRetrun(-1)

SAgent.approximateJ()
SAgent.displayExpectedRetrun(-1)

SAgent.computeActionState(1)
SAgent.displayActionState(-1)
SAgent.extractOptimalPolicy(use=True)
DAgent.displayPolicy()

DAgent.play(10, display=True)


# Derterminist Domain with MDP Domain Emulation
print("\n\nDeterministic Domain with MDP Domain Emulation : ", '\n')
DMDPAgent = Agent(DDomain, upPolicy, useMDPEmulation=True)

DMDPAgent.play(10, display=True)

DMDPAgent.computeExpectedReturn(1)
DMDPAgent.displayExpectedRetrun(-1)

DMDPAgent.approximateJ()
DMDPAgent.displayExpectedRetrun(-1)

DMDPAgent.computeActionState(1)
DMDPAgent.displayActionState(-1)

DMDPAgent.approximatQ()
DMDPAgent.displayActionState(-1)
DMDPAgent.extractOptimalPolicy(use=True)
DMDPAgent.displayPolicy()

DMDPAgent.play(10, display=True)


# Stochastic Domain with MDP Domain Emulation
print("\n\nStochastic Domain with MDP Domain Emulation : ", '\n')
SMDPAgent = Agent(SDomain, upPolicy, useMDPEmulation=True)

SMDPAgent.play(10, display=True)

SMDPAgent.computeExpectedReturn(1)
SMDPAgent.displayExpectedRetrun(-1)

SMDPAgent.approximateJ()
SMDPAgent.displayExpectedRetrun(-1)

SMDPAgent.computeActionState(1)
SMDPAgent.displayActionState(-1)

SMDPAgent.approximatQ()
SMDPAgent.displayActionState(-1)
SMDPAgent.extractOptimalPolicy(use=True)
SMDPAgent.displayPolicy()

SMDPAgent.play(10, display=True)
