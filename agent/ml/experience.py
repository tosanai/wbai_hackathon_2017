# coding: utf-8

import numpy as np
from chainer import cuda

from threading import Thread
from ml.vae import VAE

class Experience:
    def __init__(self, use_gpu=0, data_size=10**5, replay_size=32, hist_size=1, initial_exploration=10**3, dim=10240):

        self.use_gpu = use_gpu
        self.data_size = data_size
        self.replay_size = replay_size
        self.hist_size = hist_size
        # self.initial_exploration = 10
        self.initial_exploration = initial_exploration
        self.dim = dim

        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

        self.vae = VAE((hist_size, dim))
        self.d_index = 0

        self.success = np.empty(shape=(0, self.hist_size, self.dim))
        self.distance = np.empty(shape=(0, 3, self.hist_size, self.dim))

    def stock(self, time, state, action, reward, state_dash, episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = state_dash
        self.d[4][data_index] = episode_end_flag

        if episode_end_flag is True:
            # states = self.d[0][self.d_index:data_index + 1].reshape((data_index + 1 - self.d_index, self.hist_size * self.dim))
            # self.vae.learn(states)
            # learn_thread = Thread(target=self.vae.learn, kwargs={'x_train': states})
            # learn_thread.start()
            # self.d_index = data_index + 1
            # z = self.vae.encode(state)
            self.success = np.append(self.success, state, axis=0)
        else:
            self.distance[action] = np.append(self.distance[action], np.abs(state - state_dash), axis=0)

    def replay(self, time):
        replay_start = False
        if self.initial_exploration < time:
            replay_start = True
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in range(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)

            if self.success.size != 0:
                success_mean = np.mean(self.success, axis=0)
                # success_distance = np.sum(success_mean - self.d[3][-1])
            else:
                success_distance = None
            action_distance = []
            for e in self.distance:
                if e.size != 0:
                    action_distance.append(np.mean(e, axis=0))
                else:
                    action_distance.append(None)

            return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay,\
                    success_distance, action_distance

        else:
            return replay_start, 0, 0, 0, 0, False

    def end_episode(self, time, last_state, action, reward):
        self.stock(time, last_state, action, reward, last_state, True)
        replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay = \
            self.replay(time)

        return replay_start, s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay

    def vae_learn(self, x):
        self.vae.learn(x)
