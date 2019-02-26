import numpy as np
from src.Domain import Domain


class Agent:

    def __init__(self, domain: Domain, policy):
        self.domain = domain
        self.policy = policy
        self.initial_policy = policy
        x_max, y_max = self.domain.get_shape()
        self.expected_return_matrix = [np.array([[0. for _ in range(x_max)] for _ in range(y_max)])]
        self.action_state_matrix = [np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for _ in range(x_max)] for _ in range(y_max)])]

    def get_domain(self):
        return self.domain

    def get_policy(self):
        return self.policy

    def get_expected_return_matrix(self, n=None):
        if n is None:
            return self.expected_return_matrix
        if n == -1:
            return self.expected_return_matrix[-1]
        assert len(self.expected_return_matrix) > n > 1, "Invalid index"
        return self.expected_return_matrix[n]

    def get_action_state_matrix(self, n=None):
        if n is None:
            return self.action_state_matrix
        if n == -1:
            return self.action_state_matrix[-1]
        assert len(self.expected_return_matrix) > n > 1, "Invalid Index"
        return self.action_state_matrix[n]

    def play(self, nb_moves, position=(0, 0), display=False):
        # the origin is on the top left of the board
        # return a trajectory (ht = x0, m0, r0, x1 ... xn, mn, tn ,xn+1)
        trajectory = []
        cumulated_reward = 0
        for i in range(nb_moves):
            trajectory.append(position)
            move = self.policy(self.domain, position)
            trajectory.append(move)
            reward = self.domain.reward(position, move)
            trajectory.append(reward)
            cumulated_reward += reward
            position = self.domain.move(position, move)
            self.domain.update_noise()
            if display:
                print("Turn " + str(i))
                print("   Use move : " + str(move))
                print("   Get reward: " + str(reward))
                print("   End the turn in position : " + str(position))
        trajectory.append(position)
        if display:
            print("Cumulated Reward : " + str(cumulated_reward))
            print("End Position : " + str(position))
        return trajectory

    def expected_return_compute(self, n, state=None, erase=False):
        x_max, y_max = self.domain.get_shape()
        if erase:
            self.expected_return_matrix = [np.array([[0. for _ in range(x_max)] for _ in range(y_max)])]
        if len(self.expected_return_matrix) > n:
            if state is not None:
                return self.expected_return_matrix[n][state[0]][state[1]]
            return self.expected_return_matrix[n]
        for it in range(n+1-len(self.expected_return_matrix)):
            previous_matrix = self.expected_return_matrix[it - 1]
            self.expected_return_matrix.append(np.array([[0. for _ in range(x_max)] for _ in range(y_max)]))
            for i in range(x_max):
                for j in range(y_max):
                    move = self.policy(self.domain, (i, j))
                    self.expected_return_matrix[-1][j][i] = self.domain.expected_reward((i, j), move)
                    for new_position, probability in self.domain.expected_move((i, j), move):
                        self.expected_return_matrix[-1][j][i] += self.domain.get_gamma() * probability * previous_matrix[new_position[1]][new_position[0]]
        if state is not None:
            return self.expected_return_matrix[-1][state[0]][state[1]]
        return self.expected_return_matrix[-1]

    def expected_return_error(self, n):
        return (pow(self.domain.get_gamma(), n) * self.domain.get_max_reward()) / (1 - self.domain.get_gamma())

    def expected_return_approximate(self, error=0.1, erase=False):
        if erase:
            x_max, y_max = self.domain.get_shape()
            self.expected_return_matrix = [np.array([[0. for _ in range(x_max)] for _ in range(y_max)])]
        n = 0
        current_error = self.expected_return_error(n)
        while current_error >= error:
            n += 1
            self.expected_return_compute(n)
            current_error = self.expected_return_error(n)
        return self.expected_return_compute(n), current_error

    def expected_return_display(self, n=None, precision=0):
        print("Matrix of the Expected Return :")
        if n is None:
            a = 0
            b = len(self.expected_return_matrix)
        elif n == -1:
            a = len(self.expected_return_matrix) - 1
            b = len(self.expected_return_matrix)
        else:
            a = n
            b = n + 1
        for i in range(a, b):
            print("N = " + str(i))
            print("Error = " + str(self.expected_return_error(i)))
            with np.printoptions(precision=precision):
                print(self.expected_return_matrix[i], "\n")

    def action_state_compute(self, n, state=None, erase=False):
        x_max, y_max = self.domain.get_shape()
        if erase:
            self.action_state_matrix = [np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for _ in range(x_max)] for _ in range(y_max)])]
        if len(self.action_state_matrix) > n:
            if state is not None:
                return self.action_state_matrix[n][state[0]][state[1]]
            return self.action_state_matrix[n]
        for it in range(n+1-len(self.action_state_matrix)):
            previous_matrix = self.action_state_matrix[it - 1]
            self.action_state_matrix.append(np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for _ in range(x_max)] for _ in range(y_max)]))
            for i in range(x_max):
                for j in range(y_max):
                    for move in self.domain.VALID_ACTIONS:
                        self.action_state_matrix[-1][j][i][move] += self.domain.expected_reward((i, j), move)
                        for newPosition, probability in self.domain.expected_move((i, j), move):
                            self.action_state_matrix[-1][j][i][move] += self.domain.get_gamma() * probability * max(previous_matrix[newPosition[1]][newPosition[0]].values())
        if state is not None:
            return self.action_state_matrix[-1][state[0]][state[1]]
        return self.action_state_matrix[-1]

    def action_state_error(self, n):
        return (2 * pow(self.domain.get_gamma(), n) * self.domain.get_max_reward()) / (pow(1 - self.domain.get_gamma(), 2))

    def action_state_approximate(self, error=0.1, erase=False):
        if erase:
            x_max, y_max = self.domain.get_shape()
            self.action_state_matrix = [np.array([[{action: 0. for action in self.domain.VALID_ACTIONS} for _ in range(x_max)] for _ in range(y_max)])]
        n = 0
        current_error = self.action_state_error(n)
        while current_error >= error:
            n += 1
            self.action_state_compute(n)
            current_error = self.action_state_error(n)
        return self.action_state_compute(n), current_error

    def action_state_display(self, n=None, precision=0):
        print("Matrix of the Action-State :")
        if n is None:
            a = 0
            b = len(self.action_state_matrix)
        elif n == -1:
            a = len(self.action_state_matrix) - 1
            b = len(self.action_state_matrix)
        else:
            assert n > 0
            assert n < len(self.action_state_matrix)
            a = n
            b = n + 1
        for n in range(a, b):
            str_buffer = "["
            for j in range(len(self.action_state_matrix[n])):
                str_buffer += "[" if j == 0 else " ["
                for i in range(len(self.action_state_matrix[n][j])):
                    str_buffer += "{" if i == 0 else "  {"
                    for k in self.action_state_matrix[n][j][i].keys():
                        str_buffer += str(k) + ": " + str(
                            np.floor(self.action_state_matrix[n][j][i][k] * (10 ** precision)) / (10 ** precision))
                        if k != list(self.action_state_matrix[n][j][i].keys())[-1]:
                            str_buffer += "  "
                    str_buffer += "}\n" if i != len(self.action_state_matrix[n][j]) - 1 else "}"
                str_buffer += "]\n"
            print(str_buffer + "]", "\n")

    def policy_mu(self, domain, position):
        return max(self.action_state_matrix[-1][position[1]][position[0]], key=self.action_state_matrix[-1][position[1]][position[0]].get)

    def policy_get_mu(self, error=0.0001, use=False):
        n = len(self.action_state_matrix)
        if self.action_state_error(n) > error:
            self.action_state_approximate(error)
        if use:
            self.policy = self.policy_mu
        return self.policy_mu

    def policy_restore_initial(self):
        self.policy = self.initial_policy

    def policy_display(self):
        print('Current Policy :')
        x_max, y_max = self.domain.get_shape()
        visualisation = [[self.policy(self.domain, (i, j)) for i in range(x_max)] for j in range(y_max)]
        for j in range(y_max):
            print(visualisation[j])
        print()
