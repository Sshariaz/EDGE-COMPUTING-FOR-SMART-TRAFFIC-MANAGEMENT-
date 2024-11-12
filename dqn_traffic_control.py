import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import traci  # SUMO Traffic Control Interface

# Define the DQN Agent
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
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Runs SUMO simulation and trains the DQN agent while recording vehicle counts
def run_simulation(agent, batch_size, time_of_day):
    sumo_binary = "sumo-gui"  # Use 'sumo' for command-line
    sumo_cfg = "MODEL2sumo.sumocfg"  # Ensure this file is correctly set

    traci.start([sumo_binary, "-c", sumo_cfg])
    step = 0
    vehicle_counts = []
    time_intervals = []

    try:
        while step < 36:  # Simulate for 36 steps for each time of day
            traci.simulationStep()
            
            # Get state, reward, and next state
            state = np.random.rand(1, 4)  # Example, replace with real state data
            action = agent.act(state)
            
            # Traffic light control action based on the DQN output
            if action == 0:
                traci.trafficlight.setPhase("J0", 0)
            elif action == 1:
                traci.trafficlight.setPhase("J0", 1)
            elif action == 2:
                traci.trafficlight.setPhase("J0", 2)
            elif action == 3:
                traci.trafficlight.setPhase("J0", 3)

            next_state = np.random.rand(1, 4)  # Replace with actual state update logic
            reward = random.uniform(0, 1)  # Replace with computed reward logic
            done = step >= 35  # Ends after 36 steps for each rush period

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Record the number of cars at each step
            vehicle_count = len(traci.vehicle.getIDList())
            vehicle_counts.append(vehicle_count)
            time_intervals.append(step)

            print(f"{time_of_day} Step #{step}, Action: {action}, Reward: {reward}, Vehicle Count: {vehicle_count}")
            step += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        traci.close()

    # Plotting the vehicle count vs time
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, vehicle_counts, label=f"Vehicle Count - {time_of_day}", marker='o')
    plt.xlabel("Time (steps)")
    plt.ylabel("Number of Vehicles")
    plt.title(f"Vehicle Count vs. Time ({time_of_day})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main
if __name__ == "__main__":
    state_size = 4
    action_size = 4
    batch_size = 32
    agent = DQNAgent(state_size, action_size)
    
    print("\nStarting Morning Rush Simulation")
    run_simulation(agent, batch_size, "morning_rush")

    print("\nStarting Night Rush Simulation")
    run_simulation(agent, batch_size, "night_rush")
