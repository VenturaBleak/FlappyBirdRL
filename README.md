# Reinforcement Learning: Flappy Bird RL

## Overview
This repository contains implementations of Deep Q-learning to solve the Flappy Bird game using the custom `flappy_bird_gym` environment. 

### Environment: Flappy Bird

Flappy Bird is a game where the player controls a bird, attempting to fly between columns of green pipes without hitting them.

![Flappy Bird](https://github.com/YourGithubUsername/flappy_bird_rl/blob/master/flappy_bird_image_here.png)

The custom environment can be found at: `Lib/site-packages/flappy_bird_gym/envs` and is based on this [repository](https://github.com/Talendar/flappy-bird-gym/).

## Code Structure

### Files in the Repository:

**Scripts:**<br>
1. **train_pretrain.py**: Script to pre-train the agent on the Flappy Bird environment using Deep Q-learning.
2. **train_finetune.py**: Script to further train (finetune) the agent using the saved pre-trained weights.
3. **eval.py**: Script to evaluate the trained agent's performance in the Flappy Bird environment.
4. **visualize.py**: Script to visualize random actions taken by the agent.

<br>

**Trained Models:**<br>
5. **dqn_flappybird_1.zip**: Pre-trained model weights for the initial training phase. Learning to fly.
6. **dqn_flappybird_finetuned.zip**: Finetuned model weights after additional training. Learning to pass pipes.

### How to Run:

1. **Pre-train the model**:
    ```bash
    python train_pretrain.py --learning_rate 0.0005
    ```

2. **Finetune the model**:  
    ```bash
    python train_finetune.py
    ```

3. **Evaluate the trained model**:  
    ```bash
    python eval.py
    ```

4. **Visualize random agent actions**:  
    ```bash
    python visualize.py
    ```
