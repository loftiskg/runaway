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

    def move(self, move, lookahead=False):
        self.hit_wall = False
        if move == LEFT:
            return self._move_left(lookahead)
        elif move == RIGHT:
            return self._move_right(lookahead)
        elif move == UP:
            return self._move_up(lookahead)
        elif move == DOWN:
            return self._move_down(lookahead)

    def _move_right(self, lookahead=False):
        x, y = self.pos.topleft
        center = self.pos.center
        if (self.pos.right + self.speed) <= WIDTH :
            center = self.pos.centerx+self.speed, self.pos.centery
            if not lookahead:
                self.pos = self.image.get_rect().move(x + self.speed, y)
        elif not lookahead:
            self.hit_wall = True
        return center


    def _move_left(self, lookahead=False):
        x, y = self.pos.topleft
        center = self.pos.center
        if (self.pos.left - self.speed) >= 0:
            center = self.pos.centerx-self.speed, self.pos.centery
            if not lookahead:
                self.pos = self.image.get_rect().move(x - self.speed, y)
        elif not lookahead:
            self.hit_wall = True
        return center

    def _move_up(self, lookahead=False):
        x, y = self.pos.topleft
        center = self.pos.center
        if (self.pos.top - self.speed) >= 0:
            center = self.pos.centerx, self.pos.centery-self.speed
            if not lookahead:
                self.pos = self.image.get_rect().move(x, y - self.speed)
        elif not lookahead:
            self.hit_wall = True
        return center

    def _move_down(self, lookahead=False):
        x, y = self.pos.topleft
        center = self.pos.center
        if (self.pos.bottom + self.speed) <= HEIGHT:
            center = self.pos.centerx, self.pos.centery+self.speed
            if not lookahead:
                self.pos = self.image.get_rect().move(x, y + self.speed)
        elif not lookahead:
            self.hit_wall = True

        return center
