from collections import deque, namedtuple
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam


class ReplayBuffer(tf.Module):
    def __init__(self, batch_size, mem_size=5000):
        self.mem_size = mem_size
        self.counter = 0
        self.transition = namedtuple('transition', ('state', 'action', 'reward', 'state_', 'done'))
        self.memory = deque(maxlen=mem_size)
        self.batch_size = batch_size
        self.action_space = [(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), # (Steer, Gas, Break)
                             (-1, 1,   0), (0, 1,   0), (1, 1,   0),         
                             (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), 
                             (-1, 0,   0), (0, 0,   0), (1, 0,   0)]

    def memorize(self, state, action, reward, state_, done):
        trans = self.transition(state, action, reward, state_, done)
        self.memory.append(trans)
        self.counter += 1

    def sample_buffer(self):
        batch = random.sample(self.memory, self.batch_size)

        states  = np.array([b.state for b in batch if b is not None])
        # print(len(states))
        # print(len(states[0]))
        # print(len(states[0][0]))
        states_ = np.array([b.state_ for b in batch if b is not None])
        rewards = np.array([b.reward for b in batch if b is not None])
        actions = np.array([b.action for b in batch if b is not None])
        dones   = np.array([b.done for b in batch if b is not None])

        return states, actions, rewards, states_, dones


def build_model(n_actions, lr=.001):
        model = Sequential()
        model.add(Conv2D(filters = 8, kernel_size = 7, activation='relu', 
        input_shape=(96,96,4), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(filters = 16, kernel_size = 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(n_actions, activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr, epsilon=1e-5))
        return model


class Agent:
    def __init__(self, lr=.001, gamma=.99, tau=0, n_actions=12, epsilon=1, input_shape=(3,96,96), batch_size=64):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learn_counter = 0
        self.update_rate = float(1e3)
        self.action_space = [(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), # (Steer, Gas, Break)
                             (-1, 1,   0), (0, 1,   0), (1, 1,   0),         
                             (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), 
                             (-1, 0,   0), (0, 0,   0), (1, 0,   0)]
        # self.action_space = [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0.3)]
        self.memory = ReplayBuffer(self.batch_size, 1000000)
        self.model = build_model(n_actions)
        self.t_model = build_model(n_actions)

    def store(self, state, action, reward, state_, done):
        self.memory.memorize(state, action, reward, state_, done)
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            state = np.array([state])
            state_tensor = tf.convert_to_tensor(np.einsum('lijk->ljki', state))
            
            act_values = self.model.predict(state_tensor)
            action = np.argmax(act_values[0])
            return self.action_space[action]
        else:
            return random.sample(self.action_space, 1)[0]

    def learn(self):
        if self.memory.counter < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer()
        states_tensor = tf.convert_to_tensor(np.einsum('lijk->ljki', states))
        states__tensor = tf.convert_to_tensor(np.einsum('lijk->ljki', states_))
        
        q_target_next = np.amax(self.t_model.predict(states__tensor), axis=1)
        q_target = rewards + self.gamma * q_target_next * (1 - dones)
        q_pred = self.model.predict(states_tensor)
        
        for i, action in enumerate(actions):
            idx = self.action_space.index(tuple(action))
            q_pred[i, idx] = q_target[i]
        self.model.fit(states_tensor, q_pred, epochs=1, verbose=0)

        self.epsilon = self.epsilon - 1e-3 if self.epsilon > .01 else .01

        if self.learn_counter % self.update_rate == 0:
            self.soft_update()

    def soft_update(self):
        weights = self.model.get_weights()
        target_weights = self.t_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.t_model.set_weights(target_weights)

    def save_model(self, name):
        print('saving agent...')
        self.model.save(name)

    def load_model(self, name):
        print('loading agent...')
        self.model = load_model(name)