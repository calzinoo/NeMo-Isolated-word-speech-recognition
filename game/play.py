import subprocess
import sys
import time

def main():
    print("=======================================")
    print("   AVVIO MOTORE VOCALE (NEURAL NET)    ")
    print("=======================================")
    
    # Avvia il NUOVO script vocale dedicato al gioco
    ai_process = subprocess.Popen([sys.executable, "game_inference.py"])
    
    # Diamogli 10 secondi per caricare PyTorch e NeMo nella scheda video
    print("Attendo 10 secondi per il caricamento del modello NeMo...")
    time.sleep(20)
    
    print("\n=======================================")
    print("           AVVIO DEL GIOCO             ")
    print("=======================================")
    
    # --- LA MODIFICA È QUI ---
    # cwd="game" dice al terminale di spostarsi nella cartella 'game' 
    # PRIMA di lanciare main.py, così trova tutti gli import e le classi!
    game_process = subprocess.Popen([sys.executable, "main.py"])
    
    # Mette in pausa questo launcher finché giochi
    game_process.wait()
    
    # Quando chiudi il gioco, spegne anche il microfono!
    print("\nGioco terminato. Spegnimento del microfono e dell'AI...")
    ai_process.terminate()
    print("Chiusura completata. Alla prossima!")

if __name__ == "__main__":
    main()