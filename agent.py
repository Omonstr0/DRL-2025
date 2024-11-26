# agent.py
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from game_functions import can_place
from collections import deque
from experience_replay import ExperienceReplay

class Agent:

    def __init__(self, player_id):
        self.player_id = player_id

    def select_tile(self, deck, discard_pile):
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes.")

    def place_tile(self, grids, tile, discard_pile, GRID_SIZE):
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes.")
###############################
class DQLearning(Agent):
    def __init__(self, player_id, state_size=16, action_size=16, epsilon=1.0, epsilon_decay=0.995, epsilon_end=0.01,
                 gamma=0.99):
        super().__init__(player_id)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu', input_shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def get_action(self, state, valid_actions):
        # Sélection d'action epsilon-greedy avec masquage des actions invalides
        if np.random.rand() <= self.epsilon:
            # Exploration : choisir une action valide au hasard
            action = np.random.choice(valid_actions)
        else:
            # Exploitation : utiliser le modèle pour prédire les valeurs Q
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            # Masquer les actions invalides
            invalid_actions = [a for a in range(self.action_size) if a not in valid_actions]
            q_values[invalid_actions] = -np.inf
            # Sélectionner l'action avec la valeur Q maximale
            action = np.argmax(q_values)
            # Si aucune action valide n'est trouvée, choisir au hasard
            if action not in valid_actions:
                action = np.random.choice(valid_actions)
        # Décroître epsilon pour réduire l'exploration au fil du temps
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action

    def learn(self, experiences):
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        # Prédire les valeurs Q actuelles
        current_q_values = self.model.predict(states, verbose=0)

        # Prédire les valeurs Q pour les états suivants
        next_q_values = self.model.predict(next_states, verbose=0)

        # Mettre à jour les valeurs cibles
        target_q_values = current_q_values.copy()

        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Entraîner le modèle
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

    def load_model(self, file_path):
        self.model = load_model(file_path)

    def save_model(self, file_path):
        self.model.save(file_path)

    def select_tile(self, deck, discard_pile):
        if discard_pile and random.choice([True, False]):
            tile = random.choice(discard_pile)
            discard_pile.remove(tile)
            return tile
        elif deck:
            return deck.pop()
        else:
            return None

    def place_tile(self, grids, tile, discard_pile, GRID_SIZE):
        positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
        random.shuffle(positions)
        for i, j in positions:
            if can_place(grids, self.player_id, tile, i, j, GRID_SIZE):
                # Si la case est occupée, échanger
                if grids[self.player_id][i][j] is not None:
                    discard_pile.append(grids[self.player_id][i][j])
                grids[self.player_id][i][j] = tile
                return True
        # Si aucun emplacement valide, retourner False
        return False


#####################################
class DoubleDeepQLearning(Agent):
    def __init__(self, player_id, state_size=16, action_size=16, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 gamma=0.99, learning_rate=0.001, batch_size=32, memory_size=2000, target_update_freq=5):
        super().__init__(player_id)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def update_target_model(self):
        # Mettre à jour les poids du réseau cible
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Stocker l'expérience dans la mémoire
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, valid_actions):
        # Sélection d'action epsilon-greedy avec masquage des actions invalides
        if np.random.rand() <= self.epsilon:
            action = random.choice(valid_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            invalid_actions = [a for a in range(self.action_size) if a not in valid_actions]
            q_values[invalid_actions] = -np.inf
            action = np.argmax(q_values)
            if action not in valid_actions:
                action = random.choice(valid_actions)
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return  # Pas assez d'expériences pour l'apprentissage

        # Échantillonner un batch d'expériences
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Prédire les valeurs Q actuelles et futures
        q_values = self.model.predict(states, verbose=0)
        q_next_values = self.model.predict(next_states, verbose=0)
        q_target_next_values = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                # Double DQN update
                best_next_action = np.argmax(q_next_values[i])
                q_values[i][actions[i]] = rewards[i] + self.gamma * q_target_next_values[i][best_next_action]

        # Entraîner le modèle
        self.model.fit(states, q_values, epochs=1, verbose=0)

        # Décroître epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)
        self.update_target_model()

    def select_tile(self, deck, discard_pile):
        if discard_pile and random.choice([True, False]):
            tile = random.choice(discard_pile)
            discard_pile.remove(tile)
            return tile
        elif deck:
            return deck.pop()
        else:
            return None

    def place_tile(self, grids, tile, discard_pile, GRID_SIZE):
        # Générer l'état actuel
        state = self.get_state(grids, tile)
        # Obtenir les actions valides
        valid_actions = self.get_valid_actions(grids, tile, GRID_SIZE)
        if not valid_actions:
            # Aucun emplacement valide, défausser la tuile
            discard_pile.append(tile)
            return False

        # Sélectionner une action
        action = self.get_action(state, valid_actions)
        # Appliquer l'action
        if action == self.action_size - 1:
            # Action de défausser
            discard_pile.append(tile)
            return False
        else:
            i = action // GRID_SIZE
            j = action % GRID_SIZE
            if grids[self.player_id][i][j] is not None:
                discard_pile.append(grids[self.player_id][i][j])
            grids[self.player_id][i][j] = tile
            return True

    def get_state(self, grids, tile_in_hand):
        grid = grids[self.player_id]
        state = []
        for row in grid:
            for cell in row:
                state.append(0 if cell is None else cell)
        state.append(tile_in_hand if tile_in_hand is not None else 0)
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self, grids, tile, GRID_SIZE):
        valid_actions = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if can_place(grids, self.player_id, tile, i, j, GRID_SIZE):
                    action_index = i * GRID_SIZE + j
                    valid_actions.append(action_index)
        # Ajouter l'action de défausse
        valid_actions.append(self.action_size - 1)  # Index pour défausser
        return valid_actions

#############################
class DoubleDeepQLearningWithExperienceReplay(Agent):
    def __init__(self, player_id, state_size=17, action_size=17, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 gamma=0.99, learning_rate=0.001, batch_size=64, memory_size=2000, target_update_freq=5):
        super().__init__(player_id)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon  # Taux d'exploration initial
        self.epsilon_decay = epsilon_decay  # Décroissance du taux d'exploration
        self.epsilon_min = epsilon_min  # Taux d'exploration minimum
        self.gamma = gamma  # Facteur de réduction
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq  # Fréquence de mise à jour du réseau cible
        self.memory = ExperienceReplay(capacity=memory_size, batch_size=batch_size)
        self.model = self.build_model()  # Réseau principal
        self.target_model = self.build_model()  # Réseau cible
        self.update_target_model()
        self.steps = 0  # Compteur de pas pour la mise à jour du réseau cible

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def update_target_model(self):
        # Mettre à jour les poids du réseau cible
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add_experience(state, action, reward, next_state, done)

    def get_action(self, state, valid_actions):
        # Sélection d'action epsilon-greedy avec masquage des actions invalides
        if np.random.rand() <= self.epsilon:
            action = random.choice(valid_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            invalid_actions = [a for a in range(self.action_size) if a not in valid_actions]
            q_values[invalid_actions] = -np.inf
            action = np.argmax(q_values)
            if action not in valid_actions:
                action = random.choice(valid_actions)
        return action

    def learn(self):
        if not self.memory.can_provide_sample():
            return  # Pas assez d'expériences pour l'apprentissage

        # Échantillonner un mini-batch d'expériences
        minibatch = self.memory.sample_batch()
        states = np.array([experience.state for experience in minibatch])
        actions = np.array([experience.action for experience in minibatch])
        rewards = np.array([experience.reward for experience in minibatch])
        next_states = np.array([experience.next_state for experience in minibatch])
        dones = np.array([experience.done for experience in minibatch])

        # Prédire les valeurs Q pour les états actuels et suivants
        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                # Double DQN Update
                best_next_action = np.argmax(q_next[i])
                q_values[i][actions[i]] = rewards[i] + self.gamma * q_target_next[i][best_next_action]

        # Entraîner le modèle principal
        self.model.fit(states, q_values, epochs=1, verbose=0)

        # Décroître epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Mettre à jour le réseau cible périodiquement
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)
        self.update_target_model()

    def select_tile(self, deck, discard_pile):
        # Choisir une tuile du deck ou de la défausse
        if discard_pile and random.choice([True, False]):
            tile = random.choice(discard_pile)
            discard_pile.remove(tile)
            return tile
        elif deck:
            return deck.pop()
        else:
            return None

    def place_tile(self, grids, tile, discard_pile, GRID_SIZE):
        # Générer l'état actuel
        state = self.get_state(grids, tile)
        # Obtenir les actions valides
        valid_actions = self.get_valid_actions(grids, tile, GRID_SIZE)
        if not valid_actions:
            # Aucun emplacement valide, défausser la tuile
            discard_pile.append(tile)
            return False

        # Sélectionner une action
        action = self.get_action(state, valid_actions)
        # Appliquer l'action
        if action == self.action_size - 1:
            # Action de défausser
            discard_pile.append(tile)
            next_state = self.get_state(grids, 0)
            self.remember(state, action, -1, next_state, False)
            return False
        else:
            i = action // GRID_SIZE
            j = action % GRID_SIZE
            if grids[self.player_id][i][j] is not None:
                discard_pile.append(grids[self.player_id][i][j])
            grids[self.player_id][i][j] = tile
            next_state = self.get_state(grids, 0)
            # Vérifier si le jeu est terminé pour cet agent
            done = all(grids[self.player_id][x][y] is not None for x in range(GRID_SIZE) for y in range(GRID_SIZE))
            reward = 10 if done else 1
            self.remember(state, action, reward, next_state, done)
            return True

    def get_state(self, grids, tile_in_hand):
        grid = grids[self.player_id]
        state = []
        for row in grid:
            for cell in row:
                state.append(0 if cell is None else cell)
        state.append(tile_in_hand if tile_in_hand is not None else 0)
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self, grids, tile, GRID_SIZE):
        valid_actions = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if can_place(grids, self.player_id, tile, i, j, GRID_SIZE):
                    action_index = i * GRID_SIZE + j
                    valid_actions.append(action_index)
        # Ajouter l'action de défausse
        valid_actions.append(self.action_size - 1)  # Index pour défausser
        return valid_actions

class RandomAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id)

    def get_action(self, state, valid_actions):
        return random.choice(valid_actions)

# Fonction pour obtenir la classe d'agent en fonction du nom
def get_agent_class(agent_name):
    agents = {
        "RandomAgent": RandomAgent,
        "DQLearning": DQLearning,
        "DoubleDeepQLearning": DoubleDeepQLearning,
        "DoubleDeepQLearningWithExperienceReplay": DoubleDeepQLearningWithExperienceReplay,
    }
    return agents.get(agent_name)