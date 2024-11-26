# gui.py

import pygame
import sys
import game_humain_agent  # Assurez-vous que ce fichier existe
import game_agent_agent  # Si vous avez ce fichier pour le mode Agent vs Agent

# Initialisation de Pygame
pygame.init()

# Définition des paramètres de la fenêtre
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Lucky Numbers - Menu Principal")

# Couleurs
WHITE = (255, 255, 255)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)

# Polices
FONT = pygame.font.Font(None, 36)

def main_menu():
    running = True
    while running:
        screen.fill(WHITE)
        # Titre
        title_text = FONT.render("Lucky Numbers", True, BLACK)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 30))

        # Boutons du menu principal
        BUTTON_WIDTH = 250
        BUTTON_HEIGHT = 60
        button_human_vs_agent = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, 100, BUTTON_WIDTH, BUTTON_HEIGHT)
        button_agent_vs_agent = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, 200, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Bouton Humain vs Agent
        pygame.draw.rect(screen, BLUE, button_human_vs_agent)
        text = FONT.render("Humain vs Agent", True, WHITE)
        text_rect = text.get_rect(center=button_human_vs_agent.center)
        screen.blit(text, text_rect)

        # Bouton Agent vs Agent
        pygame.draw.rect(screen, BLUE, button_agent_vs_agent)
        text = FONT.render("Agent vs Agent", True, WHITE)
        text_rect = text.get_rect(center=button_agent_vs_agent.center)
        screen.blit(text, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_human_vs_agent.collidepoint(event.pos):
                    game_humain_agent.main()
                elif button_agent_vs_agent.collidepoint(event.pos):
                    agent_vs_agent_menu()

def agent_vs_agent_menu():
    running = True
    while running:
        screen.fill(WHITE)
        # Titre
        title_text = FONT.render("Agent vs Agent", True, BLACK)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 30))

        # Boutons du sous-menu Random vs Random
        BUTTON_WIDTH = 250
        BUTTON_HEIGHT = 60
        button_simulation = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, 150, BUTTON_WIDTH, BUTTON_HEIGHT)
        button_play = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, 250, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Bouton Simulation
        pygame.draw.rect(screen, BLUE, button_simulation)
        text = FONT.render("Simulation", True, WHITE)
        text_rect = text.get_rect(center=button_simulation.center)
        screen.blit(text, text_rect)

        # Bouton Random vs Random
        pygame.draw.rect(screen, BLUE, button_play)
        text = FONT.render("Jouer", True, WHITE)
        text_rect = text.get_rect(center=button_play.center)
        screen.blit(text, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_simulation.collidepoint(event.pos):
                    game_agent_agent.simulation()
                elif button_play.collidepoint(event.pos):
                    game_agent_agent.main()

    # Retour au menu principal après la simulation ou le jeu
    main_menu()

if __name__ == "__main__":
    main_menu()