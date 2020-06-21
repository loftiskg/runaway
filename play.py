from argparse import ArgumentParser
import sys
from itertools import count

import pygame

from src.agent import TD_Agent, Random_Agent, Human_Agent
from src.constants import *
from src.game import Game


def run(agent_type, model_path, verbose):
    game = Game(randomize_start_pos=False)

    if agent_type == "td":
        my_agent = TD_Agent()
        my_agent.load_model(model_path)
    
    if agent_type == "random":
        my_agent = Random_Agent(STATE_SIZE, ACTION_SIZE)

    if agent_type == "human":
        my_agent = Human_Agent()

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)

    state = game.reset()
    move = HALT
    for i in count():
        # get move from keyboard
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                sys.exit()
        
        if agent_type == 'human':
            move = move if my_agent.act(events) is None else my_agent.act(events)
        else:
            move = my_agent.act(state)
        state, reward, done, _ = game.step(move)

        if verbose:
            print(state, reward, done,move)
        if done:
            break

        screen.fill(BLACK)
        screen.blit(game.player.image, game.player.pos)
        screen.blit(game.minion.image, game.minion.pos)
        pygame.display.update()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--agent", choices=["human", "td", 'random'], default='human')
    parser.add_argument("--agent_model_path", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    agent_type = args.agent
    agent_model_path = args.agent_model_path
    verbose = args.verbose

    if agent_model_path is None and args == "dqn":
        raise ValueError("Must specify the path to model if playing in agent mode")

    run(agent_type, agent_model_path, verbose)
