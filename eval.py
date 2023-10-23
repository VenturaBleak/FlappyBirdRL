import flappy_bird_gym
import time
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = flappy_bird_gym.make("FlappyBird-v0")

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)

# Load the trained agent
model = DQN.load("dqn_flappybird_finetuned.zip", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Enjoy trained agent
for _ in range(100):
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)

        # Processing:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward: {reward}")

        env.render()
        time.sleep(1 / 300)  # FPS

        if terminated:
            break

env.close()