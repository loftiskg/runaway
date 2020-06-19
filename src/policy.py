from src.constants import *
from src.utils import distance

# state will be defined as (target.pos.center_x target.pos.center_y )


class MinionPolicy:
    def get_move(self, state):
        target_x, target_y, minion_x, minion_y, _ = state
        current_distance = distance(target_x, minion_x, target_y, minion_y)

        moves = []
        moves.append((LEFT, distance(target_x, minion_x - 1, target_y, minion_y)))
        moves.append((RIGHT, distance(target_x, minion_x + 1, target_y, minion_y)))
        moves.append((UP, distance(target_x, minion_x, target_y, minion_y - 1)))
        moves.append((DOWN, distance(target_x - 1, minion_x, target_y, minion_y + 1)))

        return min(moves, key=lambda x: x[1])[0]


class DQNPolicy:
    def __init__(
        self, epsilon,
    ):
        self.epsilon = epsilon
