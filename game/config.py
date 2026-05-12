# config.py
import os

# --- PERCORSI CARTELLE ---
BASE_DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
MANIFEST_DIR = os.path.join(BASE_DATA_DIR, "manifests")
MODEL_DIR = "models"

# File specifici
TRAIN_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.json")
TEST_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.json") # In futuro punta a test_manifest.json
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dino_finetuned.nemo")

# --- PARAMETRI AUDIO ---
SAMPLE_RATE = 16000
TARGET_LEN_MS = 1500  # 1.5 secondi
CHUNK_SIZE = 1024

# --- PARAMETRI MODELLO E TRAINING ---
# Qui aggiungiamo i tuoi nuovi comandi!
MY_LABELS = ["destra", "sinistra", "salta", "striscia", "spacca", "background"]
PRETRAINED_MODEL = "commandrecognition_en_matchboxnet3x1x64_v2"

BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 0.005

# --- PARAMETRI INFERENZA LIVE ---
LIVE_THRESHOLD = 0.85          # Sicurezza minima per attivare il comando
LIVE_WINDOW_DURATION = 1.5     # Secondi di audio analizzati ad ogni ciclo
LIVE_CHUNK_DURATION = 0.1      # Frequenza di aggiornamento (0.1s)
COOLDOWN_FRAMES = 10           # Pausa dopo un comando rilevato
