import nemo.collections.asr as nemo_asr
import torch
import sounddevice as sd
import numpy as np
import time
from colorama import Fore, Style, init

# Inizializza colori
init(autoreset=True)

# --- CONFIGURAZIONE ---
MODEL_PATH = "models/dino_finetuned.nemo" 
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1      # Quanto audio leggiamo alla volta (0.1s)
WINDOW_DURATION = 1.5     # Quanto audio "ricorda" il modello (1.5s)
THRESHOLD = 0.85          # Sicurezza minima (0-1) per accettare il comando
COOLDOWN_FRAMES = 10      # Quanti cicli ignorare dopo un comando rilevato

def main():
    # 1. Carica il Modello
    if not torch.cuda.is_available():
        print(Fore.RED + "ATTENZIONE: Sto usando la CPU. Potrebbe essere lento.")
        device = torch.device("cpu")
    else:
        print(Fore.GREEN + f"Uso GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")

    print("Caricamento modello (potrebbe volerci un attimo)...")
    try:
        model = nemo_asr.models.EncDecClassificationModel.restore_from(MODEL_PATH)
        model.eval()
        model.to(device)
    except Exception as e:
        print(Fore.RED + f"Errore caricamento modello! Hai fatto il training?\n{e}")
        return

    # Ottieni le etichette (classi) dal modello
    labels = model.cfg.labels
    print(f"Classi rilevabili: {Fore.CYAN}{labels}")

    # 2. Prepara il Buffer (Memoria Scorrevole)
    # Calcoliamo quanti 'campioni' ci stanno in 1.5 secondi
    buffer_len = int(SAMPLE_RATE * WINDOW_DURATION)
    chunk_len = int(SAMPLE_RATE * CHUNK_DURATION)
    
    # Inizializza buffer vuoto (silenzio)
    audio_buffer = np.zeros(buffer_len, dtype=np.float32)
    
    cooldown = 0

    print(Fore.YELLOW + "\n--- IN ASCOLTO (Premi Ctrl+C per uscire) ---")
    print("Parla nel microfono...")

    # 3. Loop in Tempo Reale
    try:
        # Apre il microfono
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=chunk_len) as stream:
            while True:
                # Leggi un pezzetto di audio (es. 0.1s)
                data, overflow = stream.read(chunk_len)
                
                # Converti in array piatto
                new_audio = data.flatten().astype(np.float32)

                # SCORRIMENTO (Shift):
                # 1. Sposta tutto a sinistra
                audio_buffer = np.roll(audio_buffer, -chunk_len)
                # 2. Incolla il nuovo audio alla fine
                audio_buffer[-chunk_len:] = new_audio

                # Se siamo in "pausa" dopo un comando, saltiamo l'analisi
                if cooldown > 0:
                    cooldown -= 1
                    continue

                # PREPARAZIONE DATI PER NEMO
                # NeMo vuole un tensore [Batch, Time] -> [1, 24000]
                input_signal = torch.tensor([audio_buffer], device=device)
                input_signal_length = torch.tensor([buffer_len], device=device)

                # INFERENCE (Chiediamo al modello)
                with torch.no_grad():
                    logits = model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                    probs = torch.softmax(logits, dim=-1)
                
                # Chi ha vinto?
                pred_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_idx].item()
                pred_label = labels[pred_idx]

                # FILTRO
                # Stampiamo solo se NON è background e se siamo sicuri
                if pred_label != "_background_" and pred_label != "background" and confidence > THRESHOLD:
                    print(f"{Fore.GREEN}RILEVATO: {Style.BRIGHT}{pred_label.upper()} {Fore.WHITE}(Sicurezza: {confidence:.2f})")
                    
                    # Attiva il cooldown per non ripeterlo subito
                    cooldown = COOLDOWN_FRAMES
                else:
                    # Stampa un puntino per far vedere che è vivo (opzionale)
                    print(".", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Chiusura script.")
    except Exception as e:
        print(f"\n{Fore.RED}ERRORE MICROFONO: {e}")
        print(Fore.WHITE + "Se sei su WSL2, probabilmente non vede il microfono.")

if __name__ == "__main__":
    main()