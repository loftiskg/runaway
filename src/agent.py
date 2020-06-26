import pickle
import random
import sys
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygame import K_DOWN, K_LEFT, K_RIGHT, K_UP, KEYDOWN, KEYUP

from src.constants import *


class Human_Agent:
    def act(self, event_stream):
        for event in event_stream:
            if event.type == KEYDOWN:
                if event.key == K_LEFT:  # left arrow turns left
                    return LEFT
                elif event.key == K_RIGHT:  # right arrow turns right
                    return RIGHT
                elif event.key == K_UP:  # up arrow goes up
                    return UP
                elif event.key == K_DOWN:  # down arrow goes down
                    return DOWN
            elif event.type == KEYUP:  # check for key releases
                return HALT


class Random_Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        return np.random.randint(self.action_size)

    def performance(self, env):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action  = self.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward


class QAgent:
    def __init__(self):
        self.Q = dict()
        self.initial_Q_value = 0
        self.gamma = 0.9
        self.alpha = 0.01

    def act(self, state):
        return max(
            list(range(ACTION_SIZE)),
            key=lambda x: self.Q.get((*state, x), self.initial_Q_value),
        )

    def save_model(self, path):
        with open(path, mode="wb") as f:
            pickle.dump(self.Q, f)

    def load_model(self, path):
        with open(path, mode="rb") as f:
            Q = pickle.load(f)
        self.Q = Q

    def plot_train_stats(self, path=None):
        reward_df = pd.DataFrame({"reward": self.rewards})
        moving_avg_50 = reward_df.rolling(50, min_periods=50).mean()
        plt.plot(moving_avg_50.index, moving_avg_50.reward)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward (50 episode moving average)")
        plt.ylim(top=2000)
        if path is not None:
            plt.savefig(path)

    def save_stats(self, path):
        pd.DataFrame({"reward": self.rewards}).to_csv(path)

    def performance(self, env):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action  = self.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward

    def train(self, env, episodes, epsilon, print_every, min_epsilon, epsilon_decay):
        self.rewards = []
        last_500_rewards = deque(maxlen=100)

        # for many episodes
        for e in range(episodes):
            done = False
            state = env.reset()
            total_reward = 0
            while not done:
                # epsilon greedy: behavior policy, choose action to take
                if random.random() < epsilon:
                    action = random.randint(0, ACTION_SIZE)
                else:
                    Qs = [
                        self.Q.get((*state, a), self.initial_Q_value)
                        for a in range(ACTION_SIZE)
                    ]
                    action = np.random.choice(np.argwhere(Qs == np.amax(Qs)).flatten())
                # get the Q value for current state and action chosen above
                Q_sa = self.Q.get((*state, action), self.initial_Q_value)

                # take the action and observe the reward and next state
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # do the Q-learning update
                if not done:
                    Q_sa_next_max = self.Q.get(
                        (*next_state, self.act(next_state)), self.initial_Q_value
                    )
                    update = self.alpha * ((reward + self.gamma * Q_sa_next_max) - Q_sa)
                else:
                    update = self.alpha * (reward - Q_sa)
                self.Q[(*state, action)] = (
                    self.Q.get((*state, action), self.initial_Q_value) + update
                )

                # update current state for next time step
                state = next_state
            last_500_rewards.append(total_reward)
            self.rewards.append(total_reward)
            epsilon = max(min_epsilon, epsilon*epsilon_decay)
            if (e % print_every) == 0:
                print(f"{e} | {np.mean(last_500_rewards)}")
