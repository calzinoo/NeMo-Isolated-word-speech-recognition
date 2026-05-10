import os
import sys

# Disabilita JIT per evitare crash su Windows
os.environ['NUMBA_DISABLE_JIT'] = '1'

import torch
import sounddevice as sd
import numpy as np
from colorama import Fore, Style, init
import pydirectinput

try:
    import nemo.collections.asr as nemo_asr
except Exception as e:
    print(f"Errore critico caricamento NeMo: {e}")
    sys.exit(1)

import config

init(autoreset=True)

def main():
    print(Fore.CYAN + "--- INIZIALIZZAZIONE MOTORE VOCALE (FAST MODE) ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo utilizzato: {device.upper()}")
    
    try:
        model = nemo_asr.models.EncDecClassificationModel.restore_from(config.FINAL_MODEL_PATH)
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(Fore.RED + f"Errore nel caricamento del modello: {e}")
        return

    labels = model.cfg.labels
    print(Fore.GREEN + f"Modello caricato! Etichette: {labels}")

    # --- LA MAGIA: FINESTRA SCORREVOLE (SLIDING WINDOW) ---
    buffer_len = int(config.SAMPLE_RATE * config.LIVE_WINDOW_DURATION) # Es: 1.5 secondi
    chunk_len = int(config.SAMPLE_RATE * config.LIVE_CHUNK_DURATION)   # Es: 0.1 secondi
    
    audio_buffer = np.zeros(buffer_len, dtype=np.float32)
    cooldown = 0

    print(Fore.YELLOW + "\n--- IN ASCOLTO (Riflessi pronti!) ---")

    try:
        with sd.InputStream(channels=1, samplerate=config.SAMPLE_RATE, blocksize=chunk_len) as stream:
            while True:
                # 1. Legge solo 0.1 secondi di audio
                data, overflow = stream.read(chunk_len)
                new_audio = data.flatten().astype(np.float32)

                # 2. Fa scorrere il buffer e incolla l'audio nuovo alla fine
                audio_buffer = np.roll(audio_buffer, -chunk_len)
                audio_buffer[-chunk_len:] = new_audio

                # 3. Gestione Cooldown
                if cooldown > 0:
                    cooldown -= 1
                    continue

                # 4. Inferenza immediata (10 volte al secondo!)
                input_signal = torch.tensor([audio_buffer], device=device)
                input_signal_length = torch.tensor([buffer_len], device=device)

                with torch.no_grad():
                    logits = model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                    probs = torch.softmax(logits, dim=-1)
                
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_idx].item()
                pred_label = labels[pred_idx]

                # 5. Esecuzione Comandi
                if pred_label not in ["_background_", "background"] and confidence > config.LIVE_THRESHOLD:
                    print(f"{Fore.GREEN}RILEVATO: {Style.BRIGHT}{pred_label.upper()} {Fore.WHITE}(Sicurezza: {confidence:.2f})")
                    
                    if pred_label == "salta":
                        pydirectinput.press('up')
                    elif pred_label == "striscia":
                        pydirectinput.press('down')
                    elif pred_label == "destra":
                        pydirectinput.press('right')
                    elif pred_label == "sinistra":
                        pydirectinput.press('left')
                    
                    cooldown = config.COOLDOWN_FRAMES

    except KeyboardInterrupt:
        print(Fore.RED + "\nSpegnimento motore vocale...")
    except Exception as e:
        print(Fore.RED + f"Errore microfono: {e}")

if __name__ == '__main__':
    main()