# experience_replay.py

from collections import namedtuple
import random
import numpy as np

class PrioritizedExperienceReplay:
    def __init__(self, capacity, batch_size, alpha=0.6):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha  # Exposant pour contrôler le niveau de priorité
        self.memory = []
        self.priorities = []
        self.pos = 0
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add_experience(self, state, action, reward, next_state, done, error):
        experience = self.Experience(state, action, reward, next_state, done)
        priority = (abs(error) + 1e-6) ** self.alpha  # Calcul de la priorité

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(priority)
        else:
            self.memory[self.pos] = experience
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample_batch(self, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.pos])

        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[i] for i in indices]

        # Calcul des poids d'importance
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size