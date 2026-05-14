import subprocess
import sys
import threading

def sorvegliante_ia(processo, evento_pronto):
    """
    Questo 'mini-programma' gira in background. Legge tutto quello che 
    l'IA prova a stampare e controlla se ha detto la parola magica.
    """
    for linea in iter(processo.stdout.readline, ''):
        # Stampa la riga nel terminale principale
        sys.stdout.write(linea) 
        sys.stdout.flush()
        
        if "IN ASCOLTO" in linea:
            evento_pronto.set()

def main():
    print("=======================================")
    print("   AVVIO SISTEMA VOCALE E GIOCO        ")
    print("=======================================")
    
    #semaforo virtuale
    semaforo_ia = threading.Event()
    
    ai_process = subprocess.Popen(
        [sys.executable, "-u", "game_inference.py"], 
        cwd="game",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    thread = threading.Thread(target=sorvegliante_ia, args=(ai_process, semaforo_ia))
    thread.daemon = True # Così si spegne da solo quando chiudiamo tutto
    thread.start()
    
    print("Caricamento rete neurale in corso ... (Attendi)")
    semaforo_ia.wait() 
    
    print("\n--- IA PRONTA! AVVIO MOTORE GRAFICO ---")
    
    # Avvio gioco
    game_process = subprocess.Popen([sys.executable, "engine.py"], cwd="game")
    
    # Il terminale aspetta qui finché non si chiude la finestra del gioco
    game_process.wait()
    
    print("\nSpegnimento in corso...")
    ai_process.terminate()
    print("Sessione terminata correttamente. Alla prossima!")

if __name__ == "__main__":
    main()