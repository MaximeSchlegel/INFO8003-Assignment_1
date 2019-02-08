#DEterminist = Stohastic with B=0

class DeterministicDomain:
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    VALID_ACTIONS = [UP, RIGHT, DOWN, LEFT]

    def __init__(self, board, gamma=0.99):
        self.gamma = gamma
        self.board = board  # give the reward obtain by landing on a cell
        self.y_max, self.x_max = board.shape

    def drawNoise(self):
        pass

    def move(self, position, move):
        """
        return the position of the player starting in the position given and excuting this move
        """
        x_move, y_move = move
        x_pos, y_pos = position
        x_pos = min(max(x_pos + x_move, 0), self.x_max-1)
        y_pos = min(max(y_pos + y_move, 0), self.y_max-1)
        return x_pos, y_pos

    def reward(self, position, move):
        """
        Return the reward that the player will get if he make the move given
        """
        x, y = self.move(position, move)
        return self.board[y][x]
