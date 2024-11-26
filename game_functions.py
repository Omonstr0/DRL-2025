# game_functions.py

import pygame

def draw_boards(screen, grids, current_player, last_drawn_tile, discard_pile, discard_tile_rects, winner, winner_message,
                turn_phase=None, draw_button_rect=None, select_button_rect=None, discard_button_rect=None,
                FONT=None, SMALL_FONT=None, GRID_SIZE=4, TILE_SIZE=60, MARGIN=5, COLORS=None, deck=None):
    discard_tile_rects.clear()
    screen.fill(COLORS['WHITE'])
    for player in range(2):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = (j * (TILE_SIZE + MARGIN)) + 50 + player * 450
                y = (i * (TILE_SIZE + MARGIN)) + 100
                rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, COLORS['BLACK'], rect)
                pygame.draw.rect(screen, COLORS['WHITE'], rect.inflate(-4, -4))
                if grids[player][i][j] is not None:
                    tile_color = COLORS['RED'] if player == 0 else COLORS['YELLOW']
                    text = FONT.render(str(grids[player][i][j]), True, tile_color)
                    screen.blit(text, (x + TILE_SIZE // 4, y + TILE_SIZE // 4))
                # Indiquer les cases de la diagonale initiale
                if i == j:
                    pygame.draw.rect(screen, COLORS['GOLD'], rect, 2)

    # Affiche le message "Tour du Joueur"
    player_text = f"Tour du Joueur {current_player + 1}"
    text = FONT.render(player_text, True, COLORS['BLACK'])
    screen.blit(text, (50, 20))

    # Affiche le nombre de tuiles restantes dans la pioche
    if deck is not None:
        remaining_tiles_text = f"Tuiles restantes dans la pioche : {len(deck)}"
        text = FONT.render(remaining_tiles_text, True, COLORS['BLACK'])
        screen.blit(text, (50, 400))
    else:
        remaining_tiles_text = "Pioche indisponible"
        text = FONT.render(remaining_tiles_text, True, COLORS['BLACK'])
        screen.blit(text, (50, 400))

    # Affiche la tuile piochée ou sélectionnée
    if last_drawn_tile is not None:
        drawn_text = f"Tuile en main : {last_drawn_tile}"
        text = FONT.render(drawn_text, True, COLORS['BLACK'])
        screen.blit(text, (50, 430))

    # Affiche les tuiles dans la défausse
    if discard_pile:
        discard_text = "Défausse :"
        text = SMALL_FONT.render(discard_text, True, COLORS['BLACK'])
        screen.blit(text, (50, 460))

        # Dessiner les tuiles de la défausse comme des rectangles cliquables
        for idx, tile in enumerate(discard_pile):
            x = 50 + idx * (TILE_SIZE + MARGIN)
            y = 480  # Position verticale pour les tuiles de la défausse
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, COLORS['GRAY'], rect)
            pygame.draw.rect(screen, COLORS['BLACK'], rect, 2)
            text = FONT.render(str(tile), True, COLORS['BLACK'])
            screen.blit(text, (x + TILE_SIZE // 4, y + TILE_SIZE // 4))
            discard_tile_rects.append((rect, tile))
    else:
        discard_text = "Défausse vide"
        text = SMALL_FONT.render(discard_text, True, COLORS['BLACK'])
        screen.blit(text, (50, 460))

    # Affiche le gagnant si la partie est terminée
    if winner is not None:
        text = FONT.render(winner_message, True, COLORS['BLACK'])
        screen.blit(text, (50, 550))

    # Dessiner les boutons pour "Piocher", "Prendre Défausse", "Défausser" (uniquement pour le joueur humain)
    if current_player == 0 and winner is None:
        if turn_phase == "choose":
            if draw_button_rect:
                pygame.draw.rect(screen, COLORS['BLUE'], draw_button_rect)
                screen.blit(FONT.render("Piocher", True, COLORS['WHITE']), (draw_button_rect.x + 50, draw_button_rect.y + 10))

            if discard_pile and select_button_rect:
                pygame.draw.rect(screen, COLORS['BLUE'], select_button_rect)
                screen.blit(FONT.render("Prendre Défausse", True, COLORS['WHITE']), (select_button_rect.x + 10, select_button_rect.y + 10))
        elif turn_phase == "place" and last_drawn_tile is not None:
            # Bouton "Défausser"
            if discard_button_rect:
                pygame.draw.rect(screen, COLORS['BLUE'], discard_button_rect)
                screen.blit(FONT.render("Défausser", True, COLORS['WHITE']), (discard_button_rect.x + 50, discard_button_rect.y + 10))

    pygame.display.flip()

def initialize_diagonal(grids, deck, player, GRID_SIZE=4):
    numbers = sorted([deck.pop() for _ in range(GRID_SIZE)])
    for i in range(GRID_SIZE):
        grids[player][i][i] = numbers[i]

def can_place(grids, player, num, x, y, GRID_SIZE=4):
    original_value = grids[player][x][y]
    grids[player][x][y] = num

    # Vérifier la rangée
    row_positions = sorted([i for i in range(GRID_SIZE) if grids[player][x][i] is not None])
    row_values = [grids[player][x][i] for i in row_positions]
    if not all(row_values[i] <= row_values[i + 1] for i in range(len(row_values) - 1)):
        grids[player][x][y] = original_value
        return False

    # Vérifier la colonne
    col_positions = sorted([i for i in range(GRID_SIZE) if grids[player][i][y] is not None])
    col_values = [grids[player][i][y] for i in col_positions]
    if not all(col_values[i] <= col_values[i + 1] for i in range(len(col_values) - 1)):
        grids[player][x][y] = original_value
        return False

    grids[player][x][y] = original_value
    return True

def calculate_winner(grids, GRID_SIZE=4):
    empty_counts = [sum(1 for x in range(GRID_SIZE) for y in range(GRID_SIZE) if grids[player][x][y] is None) for player in range(2)]
    min_empty = min(empty_counts)
    winners = [player for player, count in enumerate(empty_counts) if count == min_empty]
    if len(winners) == 1:
        winner = winners[0]
        winner_message = f"Joueur {winner + 1} a gagné !"
    else:
        winner = None
        winner_message = "Match nul !"
    return winner, winner_message