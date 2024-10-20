import numpy as np
import gym
from gym import spaces
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import time
import random

# Custom Environment for Key Rotation based on attack detection and dynamic key generation
class KeyRotationEnv(gym.Env):
    def __init__(self, model):
        super(KeyRotationEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # Actions: rotate key (1) or not (0)
        self.observation_space = spaces.Discrete(2)  # States: attack detected (1) or not (0)
        self.state = 0  # Initial state: no attack
        self.step_count = 0
        self.key_rotations = 0  # Track the number of key rotations
        self.model = model  # AI model for key generation
        self.current_key = self.generate_dynamic_key()  # Initial key
    
    def generate_dynamic_key(self):
        param1 = random.random()
        param2 = random.random()
        input_data = np.array([[param1, param2]])
        key_array = self.model.predict(input_data)[0]
        key = Fernet.generate_key().decode() + ''.join(map(lambda x: str(int(x * 255)), key_array))[:32]
        return key

    def reset(self):
        self.state = np.random.choice([0, 1])  # Randomly start with attack or not
        self.step_count = 0
        self.key_rotations = 0  # Reset key rotations count
        self.current_key = self.generate_dynamic_key()  # Reset the current key
        return self.state

    def step(self, action):
        self.step_count += 1
        attack_occurred = np.random.choice([0, 1])  # Simulate attack
        done = False
        reward = 0

        if action == 1:  # If rotated the key
            self.key_rotations += 1
            self.current_key = self.generate_dynamic_key()  # Generate a new dynamic key
            if attack_occurred == 1:
                reward = 10  # Correct action, attack prevented
            else:
                reward = -1  # Unnecessary key rotation
        else:  # No key rotation
            if attack_occurred == 1:
                reward = -10  # Attack was not prevented
            else:
                reward = 1  # No attack, no key rotation

        if self.step_count >= 100:  # Episode ends after 100 steps
            done = True

        self.state = attack_occurred
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# Build the DQN model
def build_model(action_space, observation_space):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + observation_space.shape))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(action_space.n))
    model.add(Activation('linear'))
    return model

# DQN Agent setup
def build_agent(model, action_space):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=action_space.n, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn

# Key generation model
def create_keygen_model():
    model = Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))  # Two input parameters
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='sigmoid'))  # Output of size 32 for the key
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Training the Key generation model
keygen_model = create_keygen_model()
X_train = np.random.rand(1000, 2)  # Random input parameters
y_train = np.random.rand(1000, 32)  # Corresponding random keys
keygen_model.fit(X_train, y_train, epochs=10, verbose=0)

# Environment setup with keygen model
env = KeyRotationEnv(keygen_model)
states = env.observation_space
actions = env.action_space

# Build and compile the DQN model
dqn_model = build_model(actions, states)
dqn = build_agent(dqn_model, actions)

# Visualization Setup
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# Data for visualization
rewards = []
rotations = []
keys = []
steps = []

# Real-time visualization function
def update_visualization(step, reward, key_rotation, current_key):
    steps.append(step)
    rewards.append(reward)
    rotations.append(key_rotation)
    keys.append(current_key)

    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

    ax[0].plot(steps, rewards, label='Rewards', color='green')
    ax[0].set_title('Rewards Over Time')
    ax[0].set_ylabel('Reward')
    ax[0].legend()

    ax[1].plot(steps, rotations, label='Key Rotations', color='blue')
    ax[1].set_title('Key Changes Over Time')
    ax[1].set_ylabel('Rotations')
    ax[1].legend()

    ax[2].plot(steps, keys, label='Current Key', color='purple')
    ax[2].set_title('Current Key Over Time')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Key ID')
    ax[2].legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # Small pause to allow for real-time updates

# Training with real-time visualization
def train_with_visualization():
    step = 0
    for episode in range(100):
        state = env.reset()
        done = False
        while not done:
            action = dqn.forward(state)  # DQN chooses the action
            state, reward, done, _ = env.step(action)
            key_rotation = 1 if action == 1 else 0
            current_key = env.current_key
            update_visualization(step, reward, key_rotation, current_key)
            dqn.backward(reward, terminal=done)  # Update the DQN with the reward
            step += 1
    dqn.save_weights('dqn_key_rotation_visualized_weights.h5f', overwrite=True)

# Start training with real-time visualization
plt.ion()  # Enable interactive mode for real-time updates
train_with_visualization()

# End real-time updates and show final plot
plt.ioff()
plt.show()
