import random
from collections import deque

import numpy as np
import torch
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from torch import nn, optim
from torch.nn import functional as F

from src.constants import *
from src.model import Model
from src.utils import distance


class Agent:
    def __init__(self, game_object, policy):
        self.game_object = game_object
        self.policy = policy

    def act(self, state):
        print(state)
        move = self.policy.get_move(state)
        print(move)
        self.game_object.move(move)

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.eval = False
#         self.model = self._build_model()

#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#                       optimizer=Adam(lr=self.learning_rate))
#         return model

#     def memorize(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if (np.random.rand() <= self.epsilon) and (self.eval == False):
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])  # returns action

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = (reward + self.gamma *
#                           np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def load(self, name):
#         self.model.load_weights(name)

#     def save(self, name):
#         self.model.save_weights(name)

# def train_agent(agent,env,episodes):
#     batch_size = 32
#     for e in range(episodes):
#         state = env.reset()
#         state = np.reshape(state, [1, STATE_SIZE])
#         done = False
#         rewards = 0
#         while not done:
#             action = agent.act(state)
#             print(action)
#             next_state, reward, done = env.step(action)
#             next_state = np.reshape(next_state, [1, STATE_SIZE])
#             agent.memorize(state, action, reward, next_state, done)
#             state = next_state
#             rewards+= reward
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#                       .format(e, episodes, reward, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)


class DQN_Agent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
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
        normlize_vector = np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, np.sqrt(WIDTH**2 + HEIGHT**2)])
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

            #if (e%500) == 0:
            self.decay_epsilon()
            self.current_episode += 1
            self.rewards_per_episode.append(total_reward)
            print(f"Episode {self.current_episode} | Reward: {total_reward} | Epsilon: {self.epsilon}")


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
