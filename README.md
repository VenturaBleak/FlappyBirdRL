# Flappy Bird RL
# Reinforcement Learning: Flappy Bird

## Overview
This repository contains an implementation of Deep Q-learning to train an agent to play the Flappy Bird game using a custom environment based on the `flappy_bird_gym` library.

### Environment: Flappy Bird

Flappy Bird is a side-scrolling game where the player controls a bird, attempting to fly between columns of green pipes without colliding. The agent must guide the bird through these gaps to earn points. The challenge is to keep the bird afloat as long as possible without making contact with the pipes or the ground.

![Flappy Bird](https://github.com/VenturaHaze/ReinforcementLearning/blob/master/yellow_bird_playing.gif)

Our custom environment can be found at: `Lib/site-packages/flappy_bird_gym/envs`. The base for this custom environment was adapted from this [repository](https://github.com/Talendar/flappy-bird-gym/).

## Code Structure

### Files in the Repository:

1. **.gitignore**: File that specifies intentionally untracked files to ignore.
2. **dqn_flappybird_1.zip**: Pre-trained model weights for the initial training phase.
3. **dqn_flappybird_finetuned.zip**: Finetuned model weights after additional training.
4. **eval.py**: Script to evaluate the trained agent's performance in the Flappy Bird environment.
5. **train_pretrain.py**: Script to pre-train the agent on the Flappy Bird environment using Deep Q-learning.
6. **train_finetune.py**: Script to further train (finetune) the agent using the saved pre-trained weights.
7. **visualize.py**: Script to visualize the agent's gameplay.

### How to Run:

1. **Pre-train the agent**:
    ```bash
    python train_pretrain.py --total_timesteps 8000000 --learning_rate 0.0005
    ```

2. **Finetune the agent** (if you have a pre-trained model and wish to further train it):
    ```bash
    python train_finetune.py
    ```

3. **Evaluate the trained agent**:
    ```bash
    python eval.py
    ```

4. **Visualize the agent's gameplay**:
    ```bash
    python visualize.py
    ```

## Credits

A big thank you to the creators and contributors of the [flappy-bird-gym repository](https://github.com/Talendar/flappy-bird-gym/) from which our custom environment was adapted. 

Please visit the above repository to understand more about the environment, and if you have improvements or issues, feel free to contribute! 

![Flappy Bird Environment](https://github.com/Talendar/flappy-bird-gym/blob/master/assets/flappy_bird_env.gif)
