# game.py

import random
import numpy as np

class LuckyNumbersEnv:
    def __init__(self, render_on=False):
        self.render_on = render_on
        self.grid_size = 4
        self.num_players = 2
        self.deck = list(range(1, 21)) * 2  # Deux jeux de tuiles de 1 à 20
        self.discard_pile = []
        self.current_player = 0
        self.grids = None
        self.last_drawn_tile = None
        self.done = False
        self.winner = None
        self.state_size = self.grid_size * self.grid_size + 1  # 16 cases de la grille + 1 tuile en main
        self.action_size = self.grid_size * self.grid_size + 1  # 16 positions possibles + action de défausser
        self.reset()

    def reset(self):
        # Réinitialiser l'état du jeu
        random.shuffle(self.deck)
        self.discard_pile = []
        self.last_drawn_tile = None
        self.current_player = 0
        self.done = False
        self.winner = None
        self.grids = [[[None for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.num_players)]
        # Initialiser les diagonales avec des tuiles triées
        for player in range(self.num_players):
            numbers = sorted([self.deck.pop() for _ in range(self.grid_size)])
            for i in range(self.grid_size):
                self.grids[player][i][i] = numbers[i]
        return self.get_state()

    def get_state(self):
        # Retourner la représentation de l'état actuel
        # Grille du joueur actuel aplatie + dernière tuile piochée
        grid = self.grids[self.current_player]
        state = []
        for row in grid:
            for cell in row:
                if cell is None:
                    state.append(0)
                else:
                    state.append(cell)
        if self.last_drawn_tile is not None:
            state.append(self.last_drawn_tile)
        else:
            state.append(0)
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self):
        # Retourner la liste des actions valides pour l'état actuel
        # Actions : 0-15 pour les positions sur la grille, 16 pour défausser
        valid_actions = []
        if self.last_drawn_tile is None:
            # Le joueur doit piocher une tuile
            return []  # Aucune action possible jusqu'à ce qu'une tuile soit piochée
        else:
            # Vérifier les positions ou la tuile peut être placée
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.can_place(self.last_drawn_tile, i, j):
                        action_index = i * self.grid_size + j
                        valid_actions.append(action_index)
            # L'action de défausser est toujours valide
            valid_actions.append(16)
        return valid_actions

    def can_place(self, tile, x, y):
        # Vérifier si la tuile peut être placée à la position (x, y)
        grid = self.grids[self.current_player]
        if grid[x][y] is not None and grid[x][y] == tile:
            # La même tuile est déjà placée ici
            return False
        original_value = grid[x][y]
        grid[x][y] = tile

        # Vérifier la ligne
        row_values = [grid[x][i] for i in range(self.grid_size) if grid[x][i] is not None]
        if not all(row_values[i] <= row_values[i + 1] for i in range(len(row_values) - 1)):
            grid[x][y] = original_value
            return False

        # Vérifier la colonne
        col_values = [grid[i][y] for i in range(self.grid_size) if grid[i][y] is not None]
        if not all(col_values[i] <= col_values[i + 1] for i in range(len(col_values) - 1)):
            grid[x][y] = original_value
            return False

        grid[x][y] = original_value
        return True

    def step(self, action, player_id):
        # Appliquer l'action et retourner next_state, reward, done
        reward = 0
        if player_id != self.current_player:
            raise Exception("Ce n'est pas le tour du joueur.")
        if self.last_drawn_tile is None:
            # Le joueur doit piocher une tuile
            if self.deck:
                self.last_drawn_tile = self.deck.pop()
            elif self.discard_pile:
                self.last_drawn_tile = self.discard_pile.pop()
            else:
                # Plus de tuiles à piocher, fin du jeu
                self.done = True
                self.winner = self.calculate_winner()
                reward = 10 if self.winner == player_id else -10
                return self.get_state(), reward, self.done
            # Pas d'action encore effectuée
            return self.get_state(), reward, self.done
        else:
            # Le joueur doit placer ou défausser la tuile
            if action == 16:
                # Défausser la tuile
                self.discard_pile.append(self.last_drawn_tile)
                self.last_drawn_tile = None
                reward = -1  # Pénalité pour défausser
            else:
                # Placer la tuile à la position spécifiée
                i = action // self.grid_size
                j = action % self.grid_size
                if self.can_place(self.last_drawn_tile, i, j):
                    grid = self.grids[player_id]
                    # Si la case est occupée, défausser l'ancienne tuile
                    if grid[i][j] is not None:
                        self.discard_pile.append(grid[i][j])
                    grid[i][j] = self.last_drawn_tile
                    self.last_drawn_tile = None
                    reward = 1  # Récompense pour un placement réussi
                    # Vérifier si le joueur a complété sa grille
                    if all(grid[x][y] is not None for x in range(self.grid_size) for y in range(self.grid_size)):
                        self.done = True
                        self.winner = player_id
                        reward = 10  # Récompense pour la victoire
                else:
                    # Action invalide
                    reward = -5  # Pénalité pour un coup invalide
            # Changer de joueur
            self.current_player = 1 - self.current_player
            return self.get_state(), reward, self.done

    def calculate_winner(self):
        # Déterminer le vainqueur en fonction du nombre de cases vides
        empty_counts = [sum(1 for x in range(self.grid_size) for y in range(self.grid_size) if self.grids[player][x][y] is None) for player in range(self.num_players)]
        min_empty = min(empty_counts)
        winners = [player for player, count in enumerate(empty_counts) if count == min_empty]
        if len(winners) == 1:
            winner = winners[0]
        else:
            winner = None  # Match nul
        return winner

    def render(self):
        # Optionnel : implémenter le rendu si nécessaire
        pass