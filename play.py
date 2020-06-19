from argparse import ArgumentParser
import sys
from itertools import count

import pygame

from src.agent import DQN_Agent
from src.constants import *
from src.game import Game


def run(agent_type, agent_model_path, verbose):
    game = Game()

    if agent_type == "agent":
        my_agent = DQN_Agent(STATE_SIZE, ACTION_SIZE)
        my_agent.load_agent_model(agent_model_path)
        my_agent.eval = True

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)

    state = game.reset()
    move = -1
    for i in count():
        # get move from keyboard
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
            # get move from agent
        if agent_type == 'agent':
            move = my_agent.act(state)
        state, reward, done = game.step(move)

        if verbose:
            print(state, reward, done, game.t)
        if done:
            break

        screen.fill(BLACK)
        screen.blit(game.player.image, game.player.pos)
        screen.blit(game.minion.image, game.minion.pos)
        pygame.display.update()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--agent", choices=["human", "agent"], default='human')
    parser.add_argument("--agent_model_path", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    agent_type = args.agent
    agent_model_path = args.agent_model_path
    verbose = args.verbose

    if agent_model_path is None and args == "agent":
        raise ValueError("Must specify the path to model if playing in agent mode")

    run(agent_type, agent_model_path, verbose)
