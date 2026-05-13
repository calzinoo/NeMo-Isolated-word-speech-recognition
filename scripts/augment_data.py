import os
import librosa
import soundfile as sf
import config

def process_and_save(y, sr, path):
    """Assicura che l'audio salvato sia esattamente lungo quanto richiesto (es. 1.0 secondi)"""
    target_samples = int(sr * config.TARGET_LEN_MS / 1000)
    # Taglia se è troppo lungo, aggiunge silenzio se è troppo corto
    y_fixed = librosa.util.fix_length(y, size=target_samples)
    sf.write(path, y_fixed, sr)

def main():
    print("--- INIZIO CLONAZIONE AUDIO (DATA AUGMENTATION) ---")
    
    # Applichiamo la mutazione SOLO ai comandi di gioco, non al background o sconosciuto
    labels = ["destra", "sinistra", "salta", "striscia", "spacca"]

    for label in labels:
        folder = os.path.join(config.RAW_DATA_DIR, label)
        if not os.path.exists(folder): 
            continue

        # Prende i file .wav ma ESCLUDE quelli già aumentati in passato
        # (questo evita che lo script entri in loop se lo lanci due volte)
        files = [f for f in os.listdir(folder) if f.endswith('.wav') and not f.startswith('aug_')]
        
        print(f"-> Mutazione di {len(files)} file nella classe: {label.upper()}...")

        for file in files:
            file_path = os.path.join(folder, file)
            # Carica il file originale
            y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)

            # 1. Pitch Acuto (+2.5 toni)
            y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2.5)
            process_and_save(y_up, sr, os.path.join(folder, f"aug_up_{file}"))

            # 2. Pitch Grave (-2.5 toni)
            y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.5)
            process_and_save(y_down, sr, os.path.join(folder, f"aug_down_{file}"))

            # 3. Più Veloce (1.2x)
            y_fast = librosa.effects.time_stretch(y, rate=1.2)
            process_and_save(y_fast, sr, os.path.join(folder, f"aug_fast_{file}"))

            # 4. Più Lento (0.85x)
            y_slow = librosa.effects.time_stretch(y, rate=0.85)
            process_and_save(y_slow, sr, os.path.join(folder, f"aug_slow_{file}"))

    print("\n--- DATA AUGMENTATION COMPLETATA! ---")
    print("Controlla le tue cartelle: i tuoi file si sono quintuplicati!")

if __name__ == "__main__":
    main()