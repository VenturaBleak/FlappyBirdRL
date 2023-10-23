import flappy_bird_gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import time
import torch

# Parse arguments
parser = argparse.ArgumentParser(description="Simulate the agent's behavior on FlappyBird prblem task.")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate.")
parser.add_argument("--initial_exploration", type=float, default=0.07, help="Initial exploration rate.")
parser.add_argument("--final_exploration", type=float, default=0.02, help="Final exploration rate.")
parser.add_argument("--exploration_fraction", type=float, default=0.8, help="Fraction of the total number of time steps during which the exploration rate is annealed.")
parser.add_argument("--buffer_size", type=int, default=0.1, help="Size of the replay buffer as a fraction of the total number of time steps.")
parser.add_argument("--batch_size", type=int, default=64, help="Number of transitions to sample from the replay buffer at each learning step.")
parser.add_argument("--total_timesteps", type=int, default=5000000, help="Total number of time steps.")
parser.add_argument("--eval_episodes", type=int, default=100, help="Total number of simulation episodes.")
parser.add_argument("-policy", type=str, default="True", help="Policy to use for action selection (random or policy).")
args = parser.parse_args()

# Create the environment
env = flappy_bird_gym.make("FlappyBird-v0")

# Specify the network architecture
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64, 64, 64, 64])

# Create the agent
model = DQN(policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            exploration_initial_eps=args.initial_exploration,
            exploration_final_eps=args.final_exploration,
            exploration_fraction=args.exploration_fraction,
            buffer_size=int(args.total_timesteps*args.buffer_size),
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            train_freq=(int(args.batch_size/8), "step"),
            seed=42,
            verbose=1)

# if dqn_flappybird_fineturned exists, load it, else load dqn_flappybird
try:
    model.load("dqn_flappybird_finetuned", env=env)
except:
    model.load("dqn_flappybird_1", env=env)

# Train the agent and save it
model.learn(total_timesteps=args.total_timesteps, log_interval=500)

# Save the agent
model.save("dqn_flappybird_finetuned")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_flappybird_finetuned", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Enjoy trained agent
for _ in range(10):
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)

        # Processing:
        obs, reward, terminated, truncated, info = env.step(action)

        # env.render()
        # time.sleep(1 / 30)  # FPS

        if terminated:
            break

env.close()