from src.constants import *


class GameObject:
    def __init__(self, image, speed, start_pos, offset):
        self.speed = speed
        self.image = image
        self.hit_wall = False

        if offset == "center":
            self.pos = image.get_rect(center=start_pos)
        elif offset == "topleft":
            self.pos = image.get_rect(topleft=start_pos)
        elif offset == "bottomright":
            self.pos = image.get_rect(bottomright=start_pos)
        else:
            ValueError(f"Invalid {offset} is an offset value")

    def move(self, move):
        self.hit_wall = False
        if move == LEFT:
            self._move_left()
        elif move == RIGHT:
            self._move_right()
        elif move == UP:
            self._move_up()
        elif move == DOWN:
            self._move_down()

    def _move_right(self):
        x, y = self.pos.topleft
        if (self.pos.right + self.speed) <= WIDTH:
            self.pos = self.image.get_rect().move(x + self.speed, y)
        else:
            self.hit_wall = True

    def _move_left(self, lookahead=False):
        x, y = self.pos.topleft
        if (self.pos.left - self.speed) >= 0:
            self.pos = self.image.get_rect().move(x - self.speed, y)
        else:
            self.hit_wall = True

    def _move_up(self):
        x, y = self.pos.topleft
        if (self.pos.top - self.speed) >= 0:
            self.pos = self.image.get_rect().move(x, y - self.speed)
        else:
            self.hit_wall = True

    def _move_down(self, lookahead=False):
        x, y = self.pos.topleft
        if (self.pos.bottom + self.speed) <= HEIGHT:
            self.pos = self.image.get_rect().move(x, y + self.speed)
        else:
            self.hit_wall = True
