from argparse import ArgumentParser
from itertools import count

import pygame

from src.agent import DQN_Agent
from src.constants import *
from src.game import Game


def run(episodes, batch_size, model_out_path, load_model_from, show_agent):
    env = Game(randomize_start_pos=True)
    agent = DQN_Agent(STATE_SIZE, ACTION_SIZE)
    if load_model_from is not None:
        agent.load_agent_model(load_model_from)

    agent.train(episodes, env, batch_size, model_out_path)

    
    if show_agent:
        env = Game(randomize_start_pos=False)
        pygame.init()
        state = env.reset()
        screen = pygame.display.set_mode(SCREEN_SIZE)
        for i in count():
            move = agent.act(state)
            state, reward, done = env.step(move)

            print(reward, done, env.t)
            if done:
                break

            screen.fill(BLACK)
            screen.blit(env.player.image, env.player.pos)
            screen.blit(env.minion.image, env.minion.pos)
            pygame.display.update()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch_size", default=64, help='Replay batch size. Default 32')
    parser.add_argument("--model_out_path", default=None)
    parser.add_argument("--load_model_from", default=None)
    parser.add_argument("--show_agent", action="store_true")

    args = parser.parse_args()

    episodes = args.episodes
    batch_size = args.batch_size
    model_out_path = args.model_out_path
    load_model_from = args.load_model_from
    show_agent = args.show_agent

    run(episodes, batch_size, model_out_path, load_model_from, show_agent)
