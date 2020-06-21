from argparse import ArgumentParser
from itertools import count

import pygame

from src.agent import TD_Agent
from src.constants import *
from src.game import Game


def run(
    episodes, epsilon, save_model,
):
    env = Game(randomize_start_pos=False)
    agent = TD_Agent()
    agent.train(env, episodes, epsilon)
    if save_model:
        agent.save_model(f"models/TD_model_e{episodes}_epsilon{epsilon}.pkl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    episodes = args.episodes
    save_model = args.save_model
    epsilon = args.epsilon

    run(episodes, epsilon, save_model)

