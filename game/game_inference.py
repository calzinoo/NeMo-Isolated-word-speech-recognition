import os
import sys

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

    buffer_len = int(config.SAMPLE_RATE * config.LIVE_WINDOW_DURATION)
    chunk_len = int(config.SAMPLE_RATE * config.LIVE_CHUNK_DURATION)
    audio_buffer = np.zeros(buffer_len, dtype=np.float32)
    
    cooldown = 0
    pred_history = []
    FRAMES_CONFERMA = 1

    print(Fore.YELLOW + "\n--- IN ASCOLTO ---")

    try:
        with sd.InputStream(channels=1, samplerate=config.SAMPLE_RATE, blocksize=chunk_len) as stream:
            while True:
                data, overflow = stream.read(chunk_len)
                new_audio = data.flatten().astype(np.float32)

                audio_buffer = np.roll(audio_buffer, -chunk_len)
                audio_buffer[-chunk_len:] = new_audio

                if cooldown > 0:
                    cooldown -= 1
                    continue

                input_signal = torch.tensor([audio_buffer], device=device)
                input_signal_length = torch.tensor([buffer_len], device=device)

                with torch.no_grad():
                    logits = model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                    probs = torch.softmax(logits, dim=-1)
                
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_idx].item()
                pred_label = labels[pred_idx]

                soglia_richiesta = config.LIVE_THRESHOLD

                if pred_label == "sinistra":
                    soglia_richiesta = 0.70

                if pred_label not in ["_background_", "background", "unknown"] and confidence > soglia_richiesta:
                    pred_history.append(pred_label)
                else:
                    pred_history.clear()

                if len(pred_history) > FRAMES_CONFERMA:
                    pred_history.pop(0)

                if len(pred_history) == FRAMES_CONFERMA and all(x == pred_history[0] for x in pred_history):
                    parola_confermata = pred_history[0]
                    print(f"{Fore.GREEN}RILEVATO E CONFERMATO: {Style.BRIGHT}{parola_confermata.upper()} {Fore.WHITE}(Sicurezza: {confidence:.2f})")
                    
                    if parola_confermata == "salta":
                        pydirectinput.press('up')
                    elif parola_confermata == "striscia":
                        pydirectinput.press('down')
                    elif parola_confermata == "destra":
                        pydirectinput.press('right')
                    elif parola_confermata == "sinistra":
                        pydirectinput.press('left')
                    elif parola_confermata == "spacca":          
                        pydirectinput.press('space')
                    
                    cooldown = config.COOLDOWN_FRAMES
                    audio_buffer.fill(0)
                    pred_history.clear() 

    except KeyboardInterrupt:
        print(Fore.RED + "\nSpegnimento motore vocale...")
    except Exception as e:
        print(Fore.RED + f"Errore microfono: {e}")

if __name__ == "__main__":
    main()