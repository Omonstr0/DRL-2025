# agent_vs_agent.py

import pygame
import random
import sys
import time
from game_functions import draw_boards, initialize_diagonal, can_place, calculate_winner
from agent import get_agent_class, DQLearning, Random, DoubleDeepQLearning

AGENT1 = "DQLearning"
AGENT2 = "Random"

def main():
    # Initialisation de Pygame
    pygame.init()

    # Définition des paramètres de la fenêtre
    SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("LuckyNumbers - Agent vs Agent")

    # Couleurs
    COLORS = {
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0),
        'RED': (200, 0, 0),
        'YELLOW': (200, 200, 0),
        'BLUE': (0, 0, 200),
        'GREEN': (0, 200, 0),
        'GRAY': (200, 200, 200),
        'GOLD': (212, 175, 55)
    }

    # Paramètres de jeu
    GRID_SIZE = 4
    TILE_SIZE = 60
    MARGIN = 5
    FONT = pygame.font.Font(None, 32)
    SMALL_FONT = pygame.font.Font(None, 24)

    # Spécifier les agents à utiliser
    agent1_name = AGENT1  # Agent pour le joueur 1
    agent2_name = AGENT2  # Agent pour le joueur 2

    # Obtenir les classes d'agent correspondantes
    Agent1Class = get_agent_class(agent1_name)
    Agent2Class = get_agent_class(agent2_name)
    if Agent1Class is None or Agent2Class is None:
        print("Agent non reconnu.")
        sys.exit()

    # Création des agents
    agents = [Agent1Class(player_id=0), Agent2Class(player_id=1)]

    # Initialisation des variables de jeu
    deck = list(range(1, 21)) * 2  # Deux jeux de 1 à 20 pour un total de 40 tuiles
    random.shuffle(deck)
    discard_pile = []
    last_drawn_tile = None
    winner = None
    winner_message = ""
    current_player = 0  # 0 pour le joueur 1, 1 pour le joueur 2

    # Deux grilles, une pour chaque joueur
    grids = [[[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(2)]

    # Liste pour stocker les rectangles des tuiles de la défausse
    discard_tile_rects = []

    # Positionnement initial avec des valeurs croissantes sur la diagonale
    initialize_diagonal(grids, deck, 0, GRID_SIZE)  # Joueur 1
    initialize_diagonal(grids, deck, 1, GRID_SIZE)  # Joueur 2

    # Boucle principale
    running = True
    while running:
        # Dessiner les plateaux
        draw_boards(
            screen, grids, current_player, last_drawn_tile, discard_pile, discard_tile_rects,
            winner, winner_message, FONT=FONT, SMALL_FONT=SMALL_FONT,
            GRID_SIZE=GRID_SIZE, TILE_SIZE=TILE_SIZE, MARGIN=MARGIN, COLORS=COLORS, deck=deck
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        if winner is None:
            pygame.time.wait(500)  # Pause pour simuler la réflexion

            agent = agents[current_player]

            # L'agent sélectionne une tuile
            last_drawn_tile = agent.select_tile(deck, discard_pile)

            if last_drawn_tile is None:
                # Plus de tuiles disponibles
                current_player = 1 - current_player
                continue

            # L'agent place la tuile
            placed = agent.place_tile(grids, last_drawn_tile, discard_pile, GRID_SIZE)
            if not placed:
                # Si l'agent ne peut pas placer la tuile, il la défausse
                discard_pile.append(last_drawn_tile)

            last_drawn_tile = None
            current_player = 1 - current_player  # Changement de joueur

            # Vérification des conditions de victoire
            for player in range(2):
                if all(grids[player][x][y] is not None for x in range(GRID_SIZE) for y in range(GRID_SIZE)):
                    winner = player
                    winner_message = f"Agent {winner + 1} a gagné !"
                    break
            else:
                if not deck and last_drawn_tile is None:
                    winner, winner_message = calculate_winner(grids, GRID_SIZE)

        pygame.display.flip()

    pygame.quit()

def simulation():
    # Cette fonction exécute autant de parties que possible en 1 seconde
    start_time = time.time()
    games_played = 0

    # Spécifier les agents à utiliser
    agent1_name = AGENT1 # Agent pour le joueur 1
    agent2_name = AGENT2  # Agent pour le joueur 2

    # Obtenir les classes d'agent correspondantes
    Agent1Class = get_agent_class(agent1_name)
    Agent2Class = get_agent_class(agent2_name)
    if Agent1Class is None or Agent2Class is None:
        print("Agent non reconnu.")
        sys.exit()

    # Définition du temps d'exécution (par exemple, 1 seconde)
    execution_time = 1.0

    while time.time() - start_time < execution_time:
        # Initialisation des variables de jeu pour chaque partie
        deck = list(range(1, 21)) * 2
        random.shuffle(deck)
        discard_pile = []
        last_drawn_tile = None
        winner = None
        current_player = 0
        grids = [[[None for _ in range(4)] for _ in range(4)] for _ in range(2)]

        # Création des agents
        agents = [Agent1Class(player_id=0), Agent2Class(player_id=1)]

        # Positionnement initial avec des valeurs croissantes sur la diagonale
        initialize_diagonal(grids, deck, 0, 4)  # Joueur 1
        initialize_diagonal(grids, deck, 1, 4)  # Joueur 2

        # Boucle de jeu pour une partie
        while True:
            agent = agents[current_player]

            # L'agent sélectionne une tuile
            last_drawn_tile = agent.select_tile(deck, discard_pile)

            if last_drawn_tile is None:
                # Plus de tuiles disponibles
                break

            # L'agent place la tuile
            placed = agent.place_tile(grids, last_drawn_tile, discard_pile, 4)
            if not placed:
                # Si l'agent ne peut pas placer la tuile, il la défausse
                discard_pile.append(last_drawn_tile)

            last_drawn_tile = None

            # Vérification des conditions de victoire
            if all(grids[current_player][x][y] is not None for x in range(4) for y in range(4)):
                winner = current_player
                break

            # Changement de joueur
            current_player = 1 - current_player

        games_played += 1

    # Calcul du nombre de parties par seconde
    end_time = time.time()
    total_time = end_time - start_time
    games_per_second = games_played / total_time

    print(f"Nombre de parties jouées en {total_time:.2f} secondes : {games_played}")
    print(f"Nombre de parties par seconde : {games_per_second:.2f}")

    # Pause pour permettre à l'utilisateur de voir les résultats
    input("Appuyez sur Entrée pour revenir au menu principal...")

if __name__ == "__main__":
    main()