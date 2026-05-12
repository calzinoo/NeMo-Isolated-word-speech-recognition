# game/main.py
import pygame
import random
import sys
import config as config
from player import Player
from obstacle import LowObstacle, HighObstacle, BusObstacle, BreakableObstacle
from coin import Coin

pygame.init()
screen = pygame.display.set_mode((config.GAME_WIDTH, config.GAME_HEIGHT))
pygame.display.set_caption("Voice Runner AI")
clock = pygame.time.Clock()

def main():
    player = Player()
    obstacles = []
    coins = [] # <--- Lista per le monete
    
    spawn_timer = 0
    coin_spawn_timer = 0 # <--- Timer separato per le monete
    score = 0
    game_over = False

    font = pygame.font.SysFont("Arial", 36)

    while True:
        screen.fill(config.COLOR_BG)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_SPACE: 
                        main() 
                else:
                    if event.key == pygame.K_LEFT and player.lane > 0:
                        player.lane -= 1
                    if event.key == pygame.K_RIGHT and player.lane < len(config.GAME_LANES) - 1:
                        player.lane += 1
                    if event.key == pygame.K_UP and player.state == "RUNNING":
                        player.state = "JUMPING"
                        player.timer = config.JUMP_DURATION
                    if event.key == pygame.K_DOWN and player.state == "RUNNING":
                        player.state = "DUCKING"
                        player.timer = config.DUCK_DURATION
                    if event.key == pygame.K_SPACE and player.state == "RUNNING": 
                        player.state = "SMASHING"                                 
                        player.timer = config.SMASH_DURATION

        if not game_over:
            player.update()

            # --- GENERAZIONE OSTACOLI ---
            spawn_timer -= 1
            if spawn_timer <= 0:
                random_lane = random.randint(0, len(config.GAME_LANES) - 1)
                ObstacleClass = random.choice([LowObstacle, HighObstacle, BusObstacle, BreakableObstacle])
                obstacles.append(ObstacleClass(random_lane))
                spawn_timer = max(30, 90 - (score // 10)) # Aumenta difficoltà

            # --- GENERAZIONE MONETE ---
            coin_spawn_timer -= 1
            if coin_spawn_timer <= 0:
                coins.append(Coin())
                # Le monete spawnano più frequentemente degli ostacoli!
                coin_spawn_timer = random.randint(20, 50)

            # --- AGGIORNAMENTO E COLLISIONE MONETE ---
            for c in coins[:]:
                c.update()
                
                # Se il giocatore tocca la moneta...
                if player.rect.colliderect(c.rect):
                    score += config.COIN_REWARD # Prendi punti!
                    coins.remove(c)                  # La moneta scompare!
                # Se la moneta esce dallo schermo...
                elif c.rect.top > config.GAME_HEIGHT:
                    coins.remove(c)

            # --- AGGIORNAMENTO E COLLISIONE OSTACOLI ---
            for obs in obstacles[:]:
                obs.update()
                
                if player.rect.colliderect(obs.rect):
                    # Se è un muro e tu stai spaccando, e il muro non è già rotto...
                    if obs.type == "BREAKABLE" and player.state == "SMASHING":
                        if getattr(obs, 'broken', False) == False:
                            obs.broken = True
                            score += 15 # Ti dà punti extra per averlo spaccato!
                    # Se il muro è già rotto, ci passi attraverso senza morire
                    elif obs.type == "BREAKABLE" and getattr(obs, 'broken', False) == True:
                        pass
                    # Logica classica per gli altri ostacoli
                    elif obs.type == "HIGH" and player.state == "DUCKING":
                        pass 
                    elif obs.type == "LOW" and player.state == "JUMPING":
                        pass 
                    else:
                        game_over = True
                
                # Punti per la sopravvivenza (10 punti ogni volta che eviti un ostacolo)
                if obs.rect.top > config.GAME_HEIGHT - 150 and not obs.passed:
                    obs.passed = True
                    score += config.SCORE_REWARD
                
                if obs.rect.top > config.GAME_HEIGHT:
                    obstacles.remove(obs)

        # --- DISEGNO A SCHERMO ---
        pygame.draw.line(screen, config.COLOR_LINE, (200, 0), (200, config.GAME_HEIGHT), 2)
        pygame.draw.line(screen, config.COLOR_LINE, (400, 0), (400, config.GAME_HEIGHT), 2)

        # Disegna gli oggetti nell'ordine giusto (prima monete, poi ostacoli, poi giocatore)
        for c in coins:
            c.draw(screen)
            
        for obs in obstacles:
            obs.draw(screen)
        
        player.draw(screen)

        score_text = font.render(f"Punti: {score}", True, config.COLOR_LINE)
        screen.blit(score_text, (10, 10))

        if game_over:
            over_text = font.render("GAME OVER! Premi SPAZIO", True, config.COLOR_LINE)
            screen.blit(over_text, (config.GAME_WIDTH//2 - over_text.get_width()//2, config.GAME_HEIGHT//2))

        pygame.display.flip()
        clock.tick(config.GAME_FPS)

if __name__ == "__main__":
    main()