import math
import sys
from argparse import ArgumentParser
from itertools import count

import numpy as np
import pygame

from src.agent import Agent, DQN_Agent
from src.constants import *
from src.game_object import GameObject
from src.policy import MinionPolicy
from src.utils import distance
import os


class Game:
    def __init__(self, randomize_start_pos=False):
        self.player_sprite = pygame.image.load(PLAYER_SPRITE_PATH)
        self.minion_sprite = pygame.image.load(MINION_SPRITE_PATH)
        self.randomize_starts_pos = randomize_start_pos
        self.t = 0
        self.max_t = 1000

    def get_state(self):
        state = np.array((*self.player.pos.center, *self.minion.pos.center))
        d = distance(state[0], state[2], state[1], state[3])
        return (*state, d)

    def get_distance(self):
        _, _, _, _, dist = self.get_state()
        return dist

    def get_reward(self):
        if self.is_victory():
            return 1000
        if self.is_caught():
            return -1000

        return (100 / self.max_t) + self.get_distance() / (
            (WIDTH ** 2 + HEIGHT ** 2) ** (1 / 2)
        )

    def is_caught(self):
        return self.player.pos.collidepoint(self.minion.pos.center)

    def is_victory(self):
        return self.t == self.max_t

    def is_done(self):
        return self.is_victory() or self.is_caught()

    def reset(self):
        if self.randomize_starts_pos:
            player_start = (np.random.randint(WIDTH), np.random.randint(HEIGHT))
            minion_start = player_start
            while (
                distance(
                    player_start[0], minion_start[0], player_start[1], minion_start[1]
                )
                < 50
            ):
                minion_start = (np.random.randint(WIDTH), np.random.randint(HEIGHT))
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

        return state, reward, done

if __name__ == "__main__":
    move = -1
    game = Game()

    my_agent = DQN_Agent(STATE_SIZE, ACTION_SIZE)

    # train_agent(my_agent, game, 100)
    my_agent.eval = False
    my_agent.train(100, game, 64)
    my_agent.eval = True


    state = game.reset()
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)

    for i in count():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            key_states = pygame.key.get_pressed()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:  # left arrow turns left
                    move = LEFT
                elif event.key == pygame.K_RIGHT:  # right arrow turns right
                    move = RIGHT
                elif event.key == pygame.K_UP:  # up arrow goes up
                    move = UP
                elif event.key == pygame.K_DOWN:  # down arrow goes down
                    move = DOWN
            elif event.type == pygame.KEYUP:  # check for key releases
                move = -1
        move = my_agent.act(state)
        state, reward, done = game.step(move)

        print(state, reward, done, game.t)
        if done:
            break

        screen.fill(BLACK)
        screen.blit(game.player.image, game.player.pos)
        screen.blit(game.minion.image, game.minion.pos)
        pygame.display.update()

