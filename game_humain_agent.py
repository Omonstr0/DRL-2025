# humain_vs_agent.py

import pygame
import random
import sys
from game_functions import draw_boards, initialize_diagonal, can_place, calculate_winner
from agent import get_agent_class
AGENT = "REINFORCE"

def main():
    # Initialisation de Pygame
    pygame.init()

    # Définition des paramètres de la fenêtre
    SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("LuckyNumbers - Humain vs Agent")

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
    BUTTON_WIDTH = 200
    BUTTON_HEIGHT = 50
    FONT = pygame.font.Font(None, 32)
    SMALL_FONT = pygame.font.Font(None, 24)

    # Spécifier l'agent à utiliser
    agent_name = AGENT  # Modifiez ce nom pour changer l'agent

    # Obtenir la classe d'agent correspondante
    AgentClass = get_agent_class(agent_name)
    if AgentClass is None:
        print(f"Agent '{agent_name}' non reconnu.")
        sys.exit()

    # Création de l'agent
    agent = AgentClass(player_id=1)

    # Initialisation des variables de jeu
    deck = list(range(1, 21)) * 2  # Deux jeux de 1 à 20 pour un total de 40 tuiles
    random.shuffle(deck)
    discard_pile = []
    last_drawn_tile = None
    selected_discard_tile = None
    winner = None
    winner_message = ""
    current_player = 0  # 0 pour l'humain, 1 pour l'agent
    turn_phase = "choose"  # Phases: choose, place, select_discard

    # Positions des boutons
    draw_button_rect = pygame.Rect(50, 600, BUTTON_WIDTH, BUTTON_HEIGHT)
    select_button_rect = pygame.Rect(300, 600, BUTTON_WIDTH, BUTTON_HEIGHT)
    discard_button_rect = pygame.Rect(550, 600, BUTTON_WIDTH, BUTTON_HEIGHT)  # Bouton "Défausser"

    # Deux grilles, une pour chaque joueur
    grids = [[[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(2)]

    # Liste pour stocker les rectangles des tuiles de la défausse
    discard_tile_rects = []

    # Positionnement initial avec des valeurs croissantes sur la diagonale
    initialize_diagonal(grids, deck, 0, GRID_SIZE)  # Joueur humain
    initialize_diagonal(grids, deck, 1, GRID_SIZE)  # Agent

    # Fonction pour gérer le clic de l'utilisateur pour choisir une action
    def handle_click(pos):
        nonlocal last_drawn_tile, selected_discard_tile, turn_phase, current_player, discard_pile

        if current_player == 0 and turn_phase == "choose":
            if draw_button_rect.collidepoint(pos) and deck:
                last_drawn_tile = deck.pop()
                turn_phase = "place"
            elif select_button_rect.collidepoint(pos) and discard_pile:
                turn_phase = "select_discard"

        elif turn_phase == "select_discard" and discard_pile:
            # Permettre au joueur de choisir une tuile spécifique dans la défausse
            for rect, tile in discard_tile_rects:
                if rect.collidepoint(pos):
                    last_drawn_tile = tile
                    discard_pile.remove(tile)
                    turn_phase = "place"
                    break

        elif turn_phase == "place" and last_drawn_tile is not None:
            # Vérifier si le joueur clique sur le bouton "Défausser"
            if discard_button_rect.collidepoint(pos):
                discard_pile.append(last_drawn_tile)
                last_drawn_tile = None
                turn_phase = "choose"
                current_player = 1  # Passer à l'agent
                return

            # Placer ou échanger la tuile sur la grille
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    x = (j * (TILE_SIZE + MARGIN)) + 50
                    y = (i * (TILE_SIZE + MARGIN)) + 100
                    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                    if rect.collidepoint(pos):
                        if can_place(grids, 0, last_drawn_tile, i, j, GRID_SIZE):
                            # Si la case est occupée, échanger
                            if grids[0][i][j] is not None:
                                discard_pile.append(grids[0][i][j])
                            grids[0][i][j] = last_drawn_tile
                            last_drawn_tile = None
                            turn_phase = "choose"
                            current_player = 1  # Passer à l'agent
                            return
                        else:
                            # Informer le joueur que le placement est invalide
                            print("Placement invalide. Veuillez choisir une autre case ou défausser la tuile.")

    # Boucle principale
    running = True
    while running:
        # Dessiner les plateaux
        draw_boards(
            screen, grids, current_player, last_drawn_tile, discard_pile, discard_tile_rects,
            winner, winner_message, turn_phase, draw_button_rect, select_button_rect,
            discard_button_rect, FONT, SMALL_FONT, GRID_SIZE, TILE_SIZE, MARGIN, COLORS, deck
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(event.pos)

        if current_player == 1 and winner is None:
            pygame.time.wait(500)  # Pause pour simuler la réflexion

            # Agent sélectionne une tuile
            last_drawn_tile = agent.select_tile(deck, discard_pile)

            if last_drawn_tile is None:
                # Si aucune tuile ne peut être piochée, passer au joueur suivant
                current_player = 0
                turn_phase = "choose"
                continue

            # Agent place la tuile
            placed = agent.place_tile(grids, last_drawn_tile, discard_pile, GRID_SIZE)
            if not placed:
                # Si l'agent ne peut pas placer la tuile, il la défausse
                discard_pile.append(last_drawn_tile)

            last_drawn_tile = None
            current_player = 0
            turn_phase = "choose"

        # Vérification des conditions de victoire
        if winner is None:
            for player in range(2):
                if all(grids[player][x][y] is not None for x in range(GRID_SIZE) for y in range(GRID_SIZE)):
                    winner = player
                    winner_message = "Le joueur a gagné !" if winner == 0 else "L'agent a gagné !"
                    break
            else:
                if not deck and last_drawn_tile is None:
                    winner, winner_message = calculate_winner(grids, GRID_SIZE)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()