import os
import sys
import traci
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# DQN Agent for Reinforcement Learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()  # Build Neural Network for the agent

    def _build_model(self):
        # Neural Network model for Deep Q-learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Returns action with the highest reward

    def replay(self, batch_size):
        # Training the agent using experiences from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=1)  # Verbose set to 1 for more visibility
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        # Save model weights with the correct filename extension
        self.model.save_weights(f"{name}.weights.h5")

# Get state function (fetch data from the detectors such as PAN 360 in your additional file)
def get_state():
    # Use traci API to gather information from detectors (PAN 360). Adjust the logic as needed for your environment.
    state = np.random.rand(1, 4)  # Placeholder; use actual state from your detectors
    return state

# Compute reward function (customize based on your setup)
def compute_reward():
    # The reward function should be based on factors such as traffic flow, waiting time, etc.
    reward = random.random()  # Placeholder; calculate the actual reward
    return reward

# Main simulation loop using SUMO and Reinforcement Learning
def run_simulation():
    traci.start(["sumo", "-c", "MODEL1sumo.sumocfg"])  # Ensure this points to your config
    state_size = 4  # Set state size according to your environment
    action_size = 3  # Number of possible traffic light actions
    agent = DQNAgent(state_size, action_size)
    batch_size = 2  # Reduced batch size for faster training
    num_steps = 5  # You can increase or decrease this for more or less steps

    try:
        for step in range(num_steps):
            traci.simulationStep()

            # Get the current state from detectors (e.g., PAN 360 from MODEL1.add.xml)
            state = get_state()

            # Agent selects an action
            action = agent.act(state)

            # Execute the action (set traffic light phase, etc.)
            if action == 0:
                traci.trafficlight.setPhase("J1", 0)  # Adjust based on your traffic setup
            elif action == 1:
                traci.trafficlight.setPhase("J1", 1)
            elif action == 2:
                traci.trafficlight.setPhase("J1", 2)

            # Get the next state and reward
            next_state = get_state()
            reward = compute_reward()

            # Check if simulation has ended
            done = step == num_steps - 1

            # Remember the experience for replay
            agent.remember(state, action, reward, next_state, done)

            # Train the agent with replay if the memory has enough data
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            print(f"Step #{step}, Action: {action}, Reward: {reward}")

            # Save the model at regular intervals (e.g., every 100 steps)
            if step % 100 == 0:
                agent.save(f"traffic_dqn_{step}")

    except KeyboardInterrupt:
        print("Simulation interrupted. Saving model and exiting...")
        agent.save(f"traffic_dqn_interrupted")
    finally:
        traci.close()

if __name__ == "__main__":
    run_simulation()
