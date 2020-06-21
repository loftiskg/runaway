import random
import sys
from collections import defaultdict, deque

import numpy as np

from src.constants import *
import pickle


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


class TD_Agent:
    def __init__(self):
        self.Q = defaultdict(int)
        self.gamma = 0.9
        self.alpha = 0.01
        self.epsilon = 0.2

    def act(self, state):
        return max(list(range(ACTION_SIZE)), key=lambda x: self.Q[(*state, x)])

    def save_model(self, path):
        with open(path, mode="wb") as f:
            pickle.dump(self.Q, f)

    def load_model(self, path):
        with open(path, mode="rb") as f:
            Q = pickle.load(f)
            assert type(Q) == defaultdict
        self.Q = Q

    def train(self, env, episodes, epsilon):
        for e in range(episodes):
            done = False
            state = env.reset()
            last_500_rewards = deque(maxlen=500)
            total_reward = 0
            while not done:
                if random.random() < epsilon:
                    action = random.randint(0, ACTION_SIZE)
                else:
                    Qs = [self.Q[(*state, a)] for a in range(ACTION_SIZE)]
                    action = np.random.choice(np.argwhere(Qs == np.amax(Qs)).flatten())
                Q_sa = self.Q[(*state, action)]
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                if not done:
                    Q_sa_next_max = self.Q[(*next_state, self.act(next_state))]
                    update = self.alpha * ((reward + self.gamma * Q_sa_next_max) - Q_sa)
                else:
                    update = self.alpha * (reward - Q_sa)
                self.Q[(*state, action)] += update
                state = next_state
            last_500_rewards.append(total_reward)
            if (e % 500) == 0:
                print(f"{e} | {np.mean(last_500_rewards)}")
