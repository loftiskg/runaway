from argparse import ArgumentParser
from itertools import count

import pygame

from src.agent import TD_Agent
from src.constants import *
from src.game import Game


def run(
    episodes,
    save_model,
    epsilon,
    print_every,
    save_plot,
    suffix,
    save_rewards,
    min_epsilon,
    epsilon_decay,
):
    env = Game(randomize_start_pos=False)
    agent = TD_Agent()
    agent.train(env, episodes, epsilon, print_every,min_epsilon,epsilon_decay)

    if save_model:
        agent.save_model(f"models/Q_model_e{episodes}_epsilon{epsilon}_{suffix}.pkl")
    if save_plot:
        agent.plot_train_stats(
            f"plots/Q_model_e{episodes}_epsilon{epsilon}_{suffix}.png"
        )
    if save_rewards:
        agent.save_stats(f"rewards/Q_e{episodes}_epsilon{epsilon}_{suffix}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument("--save_rewards", action="store_true")
    parser.add_argument("--min_epsilon", type=float, default=0)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)

    args = parser.parse_args()
    episodes = args.episodes
    save_model = args.save_model
    epsilon = args.epsilon
    print_every = args.print_every
    save_plot = args.save_plot
    suffix = args.suffix
    save_rewards = args.save_rewards
    min_epsilon = args.min_epsilon
    epsilon_decay = args.epsilon_decay

    run(
        episodes,
        save_model,
        epsilon,
        print_every,
        save_plot,
        suffix,
        save_rewards,
        min_epsilon,
        epsilon_decay,
    )
