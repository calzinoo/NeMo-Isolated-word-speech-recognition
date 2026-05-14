import pygame
import config as config

class Player:
    def __init__(self):
        self.lane = 1
        self.x = float(config.GAME_LANES[self.lane])
        self.y = config.GAME_HEIGHT - 150
        self.state = "RUNNING" 
        self.timer = 0
        self.rect = pygame.Rect(0, 0, 60, 60)
        
        # Carica le immagini
        self.images = {
            "RUNNING": pygame.transform.scale(pygame.image.load(config.PLAYER_IMG_RUN).convert_alpha(), config.PLAYER_SIZE),
            "JUMPING": pygame.transform.scale(pygame.image.load(config.PLAYER_IMG_JUMP).convert_alpha(), config.PLAYER_SIZE),
            "DUCKING": pygame.transform.scale(pygame.image.load(config.PLAYER_IMG_DUCK).convert_alpha(), (60, 45)), 
            "SMASHING": pygame.transform.scale(pygame.image.load(config.PLAYER_IMG_SMASH).convert_alpha(), config.PLAYER_SIZE)
        }
        self.current_image = self.images["RUNNING"]
        
    def update(self):
        # Ripristina lo stato normale alla fine del timer
        if self.state in ["JUMPING", "DUCKING", "SMASHING"]:
            self.timer -= 1
            if self.timer <= 0:
                self.state = "RUNNING"
                
        # SPOSTAMENTO FLUIDO 
        target_x = config.GAME_LANES[self.lane]
        # 0.25 è la velocità di scivolamento. Più è basso, più scivola lentamente
        self.x += (target_x - self.x) * 0.25  
        self.rect.centerx = int(self.x)

        self.current_image = self.images[self.state].copy()

        # Cambia forma della hitbox e posizione Y
        if self.state == "RUNNING":
            self.rect.height = 70
            self.rect.bottom = self.y
            
        elif self.state == "JUMPING":
            self.rect.height = 70
            self.rect.bottom = self.y - 100
            
        elif self.state == "DUCKING":
            self.rect.height = 30 
            self.rect.bottom = self.y
            
        elif self.state == "SMASHING":
            self.rect.height = 70
            self.rect.bottom = self.y

    def draw(self, surface):
        img_rect = self.current_image.get_rect()
        img_rect.midbottom = self.rect.midbottom
        surface.blit(self.current_image, img_rect)