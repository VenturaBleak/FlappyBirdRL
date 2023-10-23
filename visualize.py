import time
import flappy_bird_gym
import numpy as np
from tqdm import trange

env = flappy_bird_gym.make("FlappyBird-v0")
episodes = 10000  # Changed from 100 to 100000

# Pre-assigned probabilities for each action
action_probabilities = [0.92, 0.08]

# Lists to store values
h_dist_values = []
v_dist_values = []
player_y_values = []

for episode in trange(episodes):
    obs, info = env.reset()
    while True:
        # Next action, sample with a pre-assigned probability per action
        action = np.random.choice(2, p=action_probabilities)

        # Processing:
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract values from obs
        h_dist = obs[0]
        v_dist = obs[1]

        # Append to lists
        h_dist_values.append(h_dist)
        v_dist_values.append(v_dist)

        # Rendering the game:
        env.render()
        time.sleep(1 / 30)  # FPS

        # Checking if the player is still alive
        if terminated or truncated:
            break

env.close()

# Print min and max for each value
print(f"h_dist: Min = {min(h_dist_values)}, Max = {max(h_dist_values)}")
print(f"v_dist: Min = {min(v_dist_values)}, Max = {max(v_dist_values)}")
print(f"player_y: Min = {min(player_y_values)}, Max = {max(player_y_values)}")