# main.py

from game import LuckyNumbersEnv
from agent import DQLearning, RandomAgent
import numpy as np
import os

def main():
    env = LuckyNumbersEnv()
    state_size = env.state_size
    action_size = env.action_size
    agent = DQLearning(player_id=0, state_size=state_size, action_size=action_size)
    opponent = RandomAgent()
    episodes = 10  # Réduire le nombre d'épisodes pour les tests

    for episode in range(episodes):
        state = env.reset()
        done = False

        step_count = 0  # Compteur de pas dans l'épisode

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                # Aucune action possible, passer au joueur suivant
                state = env.get_state()
                continue

            if env.current_player == 0:
                # Tour de notre agent
                action = agent.get_action(state, valid_actions)
                next_state, reward, done = env.step(action, player_id=0)
                # Entraîner l'agent
                agent.learn(state, action, reward, next_state, done)
            else:
                # Tour de l'adversaire
                action = opponent.get_action(state, valid_actions)
                next_state, reward, done = env.step(action, player_id=1)
                # Vous pouvez implémenter l'apprentissage pour l'adversaire si nécessaire

            state = next_state
            step_count += 1

        print(f"Épisode {episode + 1}/{episodes} terminé en {step_count} étapes.")

    # Sauvegarder le modèle de l'agent
    if not os.path.exists('models'):
        os.makedirs('models')
    agent.save_model('models/model.h5')
    print("Modèle sauvegardé dans 'models/model.h5'.")

if __name__ == "__main__":
    main()