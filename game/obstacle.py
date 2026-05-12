# game/obstacle.py
import pygame
import config as config

# --- CLASSE MADRE ---
class Obstacle:
    def __init__(self, lane):
        self.lane = lane
        self.passed = False
        # La hitbox di base e il tipo verranno sovrascritti dalle figlie
        self.type = "BASE" 
        self.rect = pygame.Rect(0, 0, 0, 0)

    def update(self):
        # Tutti gli ostacoli scendono allo stesso modo
        self.rect.y += config.GAME_SPEED

    def draw(self, surface):
        # Questo verrà personalizzato dalle classi figlie
        pass


# --- CLASSI FIGLIE ---

class LowObstacle(Obstacle):
    def __init__(self, lane):
        super().__init__(lane) # Chiama il costruttore della Madre
        self.type = "LOW"
        self.rect = pygame.Rect(0, -60, 60, 45)
        self.rect.centerx = config.GAME_LANES[self.lane]

    def draw(self, surface):
        pygame.draw.rect(surface, config.COLOR_OBS_LOW, self.rect)
        pygame.draw.line(surface, (0, 0, 0), (self.rect.left, self.rect.centery), (self.rect.right, self.rect.centery), 4)


class HighObstacle(Obstacle):
    def __init__(self, lane):
        super().__init__(lane)
        self.type = "HIGH"
        self.rect = pygame.Rect(0, -60, 60, 25)
        self.rect.centerx = config.GAME_LANES[self.lane]

    def draw(self, surface):
        pygame.draw.rect(surface, config.COLOR_OBS_HIGH, self.rect)
        pole_width = 8
        pole_color = (150, 150, 150)
        pygame.draw.rect(surface, pole_color, (self.rect.left, self.rect.top, pole_width, 90))
        pygame.draw.rect(surface, pole_color, (self.rect.right - pole_width, self.rect.top, pole_width, 90))


class BusObstacle(Obstacle):
    def __init__(self, lane):
        super().__init__(lane)
        self.type = "BUS"
        # Il bus è lungo e spesso (100 pixel di altezza)
        self.rect = pygame.Rect(0, -120, 60, 100) 
        self.rect.centerx = config.GAME_LANES[self.lane]

    def draw(self, surface):
        # Disegna la carrozzeria gialla
        pygame.draw.rect(surface, config.COLOR_OBS_BUS, self.rect)
        
        # Dettaglio grafico: Parabrezza azzurro
        window_color = (100, 200, 255)
        pygame.draw.rect(surface, window_color, (self.rect.left + 5, self.rect.bottom - 40, 50, 20))

class BreakableObstacle(Obstacle):
    def __init__(self, lane):
        super().__init__(lane)
        self.type = "BREAKABLE"
        self.broken = False # All'inizio è intero
        self.rect = pygame.Rect(0, -60, 60, 60) # Muro quadrato
        self.rect.centerx = config.GAME_LANES[self.lane]

    def draw(self, surface):
        if not self.broken:
            # Disegna il blocco di legno/mattoni intero
            pygame.draw.rect(surface, config.COLOR_OBS_BREAKABLE, self.rect)
            # Disegna una X nera sopra per far capire al giocatore che si può spaccare
            pygame.draw.line(surface, (0, 0, 0), self.rect.topleft, self.rect.bottomright, 4)
            pygame.draw.line(surface, (0, 0, 0), self.rect.bottomleft, self.rect.topright, 4)
        else:
            # Se è rotto, disegna solo due piccoli detriti ai lati e lascia il centro libero!
            pygame.draw.rect(surface, config.COLOR_OBS_BREAKABLE, (self.rect.left, self.rect.bottom - 15, 15, 15))
            pygame.draw.rect(surface, config.COLOR_OBS_BREAKABLE, (self.rect.right - 15, self.rect.bottom - 15, 15, 15))