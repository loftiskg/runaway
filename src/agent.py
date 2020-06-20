import random
from collections import deque

import numpy as np
import torch
from pygame import K_DOWN, K_LEFT, K_RIGHT, K_UP, KEYDOWN, KEYUP, QUIT
from torch import nn, optim
from torch.nn import functional as F

from src.constants import *
from src.model import Model
from src.utils import distance

import sys

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


class DQN_Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9
        self.epsilon = 1.0  # exploration rate
        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.model = Model(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.rewards_per_episode = []
        self.current_episode = 0
        self.eval = False

    def normalize(self, state):
        normlize_vector = np.array(
            [WIDTH, HEIGHT, WIDTH, HEIGHT, np.sqrt(WIDTH ** 2 + HEIGHT ** 2)]
        )
        return (state / normlize_vector).astype(np.float32)

    def act(self, state):
        """
        agent acts with an e-greedy policy.  If self.eval is True,
        then agent will chose action with largest Q value.
        """
        if self.eval == False and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = self.normalize(state)
        act_values = self.model(torch.tensor(state)).detach().numpy()
        return np.argmax(act_values)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def load_agent_model(self, path):
        self.model = Model(self.state_size, self.action_size)
        self.model.load_state_dict(torch.load(path))

    def train(self, episodes, env, batch_size, model_out_path):
        self.eval = False
        for e in range(episodes):
            episode_rewards = 0
            state = self.normalize(env.reset())
            done = False
            total_reward = 0
            t = 0
            # start playing Lunar Lander
            while not done:
                # get action
                action = self.act(state)
                next_state, reward, done = env.step(action)
                next_state = self.normalize(next_state)
                total_reward += reward
                self.replay_buffer.add((state, action, reward, next_state, done))
                if len(self.replay_buffer) >= batch_size:
                    # vectorized replay
                    (
                        state,
                        action,
                        reward,
                        next_state_replay,
                        replay_done,
                    ) = self.replay_buffer.sample(batch_size)
                    y = torch.tensor(reward).float()
                    y += (
                        self.gamma
                        * self.model(torch.tensor(next_state_replay))
                        .detach()
                        .max(1)
                        .values
                        * (1 - done)
                    )
                    y = y.unsqueeze(1)

                    Q_exp = self.model(torch.tensor(state)).gather(
                        1, torch.tensor(action).unsqueeze(1)
                    )

                    loss = F.mse_loss(Q_exp, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                state = next_state
                if model_out_path is not None:
                    torch.save(self.model.state_dict(), model_out_path)
                t += 1

            # if (e%500) == 0:
            self.decay_epsilon()
            self.current_episode += 1
            self.rewards_per_episode.append(total_reward)
            print(f"{self.current_episode} | {total_reward}")


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def add(self, item):
        self.buffer.append(item)

    def sample(self, n):
        samples = random.sample(self.buffer, k=n)
        state = np.array([item[0] for item in samples])
        action = np.array([item[1] for item in samples])
        reward = np.array([item[2] for item in samples])
        next_state = np.array([item[3] for item in samples])
        done = np.array([item[4] for item in samples])
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
