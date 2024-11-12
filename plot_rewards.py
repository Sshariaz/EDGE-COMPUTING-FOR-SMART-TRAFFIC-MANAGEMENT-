import matplotlib.pyplot as plt

# Sample vehicle count data for DQN and Federated Learning (morning and night rush)
dqn_morning_vehicle_counts = [4, 4, 4, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 19, 19, 19, 23, 23, 23, 23, 22, 21, 21, 20, 20, 20, 20, 20, 19, 18, 18]
dqn_night_vehicle_counts = [4, 4, 4, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 23, 22, 22, 21, 21, 21, 20, 20, 20, 20, 20, 19, 19]

fed_morning_vehicle_counts = [4, 4, 4, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 20, 19, 18, 21, 21, 21, 21, 21, 20, 20, 20, 19, 19, 17, 17, 15, 15, 14]
fed_night_vehicle_counts = [4, 4, 4, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 20, 19, 19, 22, 22, 22, 22, 22, 21, 20, 20, 20, 20, 18, 18, 17, 17, 17]

# Convert steps to time in minutes (each step = 10 seconds)
time_in_minutes = [i * 10 / 60 for i in range(len(dqn_morning_vehicle_counts))]

# Creating subplots for morning and night rush comparisons
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Morning Rush Vehicle Count Comparison
ax1.plot(time_in_minutes, dqn_morning_vehicle_counts, label="DQN - Morning Rush", color="blue", marker='o')
ax1.plot(time_in_minutes, fed_morning_vehicle_counts, label="Federated Learning - Morning Rush", color="red", marker='o')
ax1.set_xlabel("Time (minutes)")
ax1.set_ylabel("Vehicle Count")
ax1.set_title("Vehicle Count Comparison: Morning Rush (DQN vs Federated)")
ax1.legend()
ax1.grid(True)

# Night Rush Vehicle Count Comparison
ax2.plot(time_in_minutes, dqn_night_vehicle_counts, label="DQN - Night Rush", color="blue", marker='o')
ax2.plot(time_in_minutes, fed_night_vehicle_counts, label="Federated Learning - Night Rush", color="red", marker='o')
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("Vehicle Count")
ax2.set_title("Vehicle Count Comparison: Night Rush (DQN vs Federated)")
ax2.legend()
ax2.grid(True)

# Display both plots
plt.tight_layout()
plt.show()
