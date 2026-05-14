# game/coin.py
import pygame
import random
import config as config

class Coin:
    def __init__(self):
        # Sceglie una corsia casuale
        self.lane = random.randint(0, len(config.GAME_LANES) - 1)
        
        #hitbox
        self.rect = pygame.Rect(0, -40, 30, 30)
        self.rect.centerx = config.GAME_LANES[self.lane]

    def update(self):
        # Scende alla stessa velocità del gioco
        self.rect.y += config.GAME_SPEED

    def draw(self, surface):
        pygame.draw.circle(surface, config.COLOR_COIN, self.rect.center, 15)
        pygame.draw.circle(surface, (200, 150, 0), self.rect.center, 10, 2)