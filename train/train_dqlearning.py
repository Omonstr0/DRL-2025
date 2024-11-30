# train/train_dql.py

from agent import DQLearning, Random
from game import LuckyNumbersEnv
import numpy as np
import os
import matplotlib.pyplot as plt

def train_dql():
    env = LuckyNumbersEnv()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQLearning(player_id=0, state_size=state_size, action_size=action_size)
    opponent = Random(player_id=1)
    episodes = 1000

    all_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                state = env.get_state()
                continue

            if env.current_player == 0:
                action = agent.get_action(state, valid_actions)
                next_state, reward, done = env.step(action, player_id=0)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
            else:
                action = opponent.get_action(state, valid_actions)
                next_state, reward, done = env.step(action, player_id=1)

            state = next_state
            total_reward += reward

        all_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Épisode {episode + 1}/{episodes}, Récompense moyenne sur 100 épisodes : {avg_reward:.2f}")

    # Sauvegarder le modèle
    if not os.path.exists('../models'):
        os.makedirs('../models')
    agent.save_model('../models/dql_model.h5')
    print("Modèle DQL sauvegardé.")

    # Visualiser les récompenses
    plt.plot(all_rewards)
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense')
    plt.title('Récompenses par épisode - DQL')
    plt.show()

if __name__ == "__main__":
    train_dql()