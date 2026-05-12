
import os

MODEL_DIR = "models"
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dino_finetuned.nemo")


# --- PARAMETRI AUDIO ---
SAMPLE_RATE = 16000
TARGET_LEN_MS = 1500  # 1.5 secondi
CHUNK_SIZE = 1024

# --- PARAMETRI INFERENZA LIVE ---
LIVE_THRESHOLD = 0.90          # Sicurezza minima per attivare il comando
LIVE_WINDOW_DURATION = 1.5     # Secondi di audio analizzati ad ogni ciclo
LIVE_CHUNK_DURATION = 0.1      # Frequenza di aggiornamento (0.1s)
COOLDOWN_FRAMES = 1           # Pausa dopo un comando rilevato


# --- PARAMETRI GIOCO (PYGAME) ---
GAME_WIDTH = 600
GAME_HEIGHT = 800
GAME_FPS = 60
GAME_SPEED = 6

# Corsie (Coordinate X del centro di ogni corsia)
GAME_LANES = [100, 300, 500]

# Dinamiche giocatore
JUMP_DURATION = 60    # Quanti frame dura il salto
DUCK_DURATION = 60    # Quanti frame dura la scivolata
SCORE_REWARD = 10     # Punti per ogni ostacolo superato
SMASH_DURATION = 50   # Quanti frame dura la modalità "spacca"

# Colori (R, G, B)
COLOR_BG = (50, 50, 50)
COLOR_LINE = (255, 255, 255)
COLOR_PLAYER_RUN = (0, 150, 255)  # Blu
COLOR_PLAYER_JUMP = (0, 255, 0)   # Verde
COLOR_PLAYER_DUCK = (150, 0, 255) # Viola
COLOR_OBS_LOW = (255, 100, 0)
COLOR_OBS_HIGH = (255, 0, 0)
COLOR_OBS_BUS = (255, 200, 0)
COLOR_PLAYER_SMASH = (255, 50, 50)  # Rosso fuoco quando spacca
COLOR_OBS_BREAKABLE = (139, 69, 19) # Marrone (muro di legno/mattoni)

# --- PARAMETRI MONETE ---
COLOR_COIN = (255, 215, 0)  # Giallo Oro
COIN_REWARD = 5             # Quanti punti dà una singola moneta