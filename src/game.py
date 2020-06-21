import math
import sys

import numpy as np
import pygame
from gym import spaces

from src.constants import *
from src.game_object import GameObject
from src.policy import MinionPolicy
from src.utils import distance

class Game:
    def __init__(self, randomize_start_pos=False):
        self.player_sprite = pygame.transform.scale(
            pygame.image.load(PLAYER_SPRITE_PATH), (int(WIDTH * 0.1), int(HEIGHT * 0.1))
        )
        self.minion_sprite = pygame.transform.scale(
            pygame.image.load(MINION_SPRITE_PATH), (int(WIDTH * 0.1), int(HEIGHT * 0.1))
        )
        self.randomize_starts_pos = randomize_start_pos
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self.observation_space = spaces.Box(low=np.array([0,0,0,0,0]),
                                            high=np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, MAX_DIST]),
                                            dtype=np.float32)
        self.t = 0
        self.max_t = 1000

    def get_state(self):
        state = np.array((*self.player.pos.center, *self.minion.pos.center))
        d = distance(state[0], state[2], state[1], state[3])
        return (*state, d)

    def get_distance(self):
        _, _, _, _, dist = self.get_state()
        return dist

    def hit_wall(self):
        return self.player.hit_wall

    def get_reward(self):
        if self.is_victory():
            return 1000
        if self.is_caught():
            return -100

        return 1

        # return (100 / self.max_t) + self.get_distance() / (
        #     (WIDTH ** 2 + HEIGHT ** 2) ** (1 / 2)
        # )

    def is_caught(self):
        return self.player.pos.collidepoint(self.minion.pos.center)

    def is_victory(self):
        return self.t == self.max_t

    def is_done(self):
        return self.is_victory() or self.is_caught()

    def reset(self):
        if self.randomize_starts_pos:
            player_start = (np.random.randint(10, WIDTH-10), np.random.randint(10,HEIGHT-10))
            minion_start = player_start
            while (
                distance(
                    player_start[0], minion_start[0], player_start[1], minion_start[1]
                )
                < 25
            ):
                minion_start = (np.random.randint(10,WIDTH-10), np.random.randint(10,HEIGHT-10))
        else:
            player_start = (0, 0)
            minion_start = (WIDTH, HEIGHT)

        self.t = 0
        self.player = GameObject(
            self.player_sprite, SPEED, start_pos=player_start, offset="topleft"
        )
        self.minion = GameObject(
            self.minion_sprite, SPEED / 4, start_pos=minion_start, offset="bottomright",
        )
        return self.get_state()

    def step(self, player_move):
        self.t += 1
        current_state = self.get_state()
        minion_move = MinionPolicy().get_move(current_state)
        self.player.move(player_move)
        self.minion.move(minion_move)

        state = self.get_state()
        reward = self.get_reward()
        done = self.is_done()

        return state, reward, done, {}
    
    def lookahead(self, player_move):
        player_pos = self.player.pos
        minion_pos = self.minion.pos

        state,reward,done,debug = self.step(player_move)
        
        self.player.pos = player_pos
        self.minion.pos = minion_pos
        return state, reward, done, debug

        
