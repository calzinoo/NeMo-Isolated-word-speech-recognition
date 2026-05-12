import pygame
import config as config


class Player:
    def __init__(self):
        self.lane = 1 # Parte nella corsia centrale (indice 1)
        self.y = config.GAME_HEIGHT - 150
        self.state = "RUNNING" # Stati: RUNNING, JUMPING, DUCKING
        self.timer = 0
        self.rect = pygame.Rect(0, 0, 60, 60)
        
    def update(self):
        # Ripristina lo stato normale alla fine del timer
        if self.state in ["JUMPING", "DUCKING"]:
            self.timer -= 1
            if self.timer <= 0:
                self.state = "RUNNING"
                
        # Aggiorna posizione X in base alla corsia
        self.rect.centerx = config.GAME_LANES[self.lane]
        self.rect.bottom = self.y

        # Cambia forma e posizione Y
        if self.state == "RUNNING":
            self.rect.height = 60
        elif self.state == "JUMPING":
            self.rect.height = 60
            self.rect.bottom = self.y - 100
        elif self.state == "DUCKING":
            self.rect.height = 30
            self.rect.bottom = self.y

    def draw(self, surface):
        color = config.COLOR_PLAYER_RUN
        if self.state == "JUMPING": color = config.COLOR_PLAYER_JUMP
        if self.state == "DUCKING": color = config.COLOR_PLAYER_DUCK
        pygame.draw.rect(surface, color, self.rect)
        