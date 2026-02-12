# scripts/create_manifest.py
import json
import os
import librosa
import argparse
from tqdm import tqdm

def create_manifest(data_dir, manifest_path):
    """
    Scansiona una cartella e crea il file JSON per NeMo.
    Struttura attesa: data_dir/nome_classe/file_audio.wav
    """
    manifest_file = open(manifest_path, 'w')
    files_found = 0
    
    # Cerca tutte le sottocartelle (che sono le nostre classi/labels)
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processando classe: {class_name}...")
        
        # Scansiona i file WAV
        for wav_file in tqdm(os.listdir(class_dir)):
            if not wav_file.endswith('.wav'):
                continue
                
            audio_path = os.path.join(class_dir, wav_file)
            
            # Calcola durata (NeMo ne ha bisogno)
            try:
                duration = librosa.get_duration(filename=audio_path)
                
                # Crea l'oggetto JSON
                entry = {
                    'audio_filepath': os.path.abspath(audio_path),
                    'duration': duration,
                    'command': class_name
                }
                
                # Scrivi nel file (una riga per JSON)
                json.dump(entry, manifest_file)
                manifest_file.write('\n')
                files_found += 1
                
            except Exception as e:
                print(f"Errore su {wav_file}: {e}")

    manifest_file.close()
    print(f"\nFinito! Creato manifest con {files_found} file in: {manifest_path}")

if __name__ == "__main__":
    # Esempio di utilizzo:
    # python scripts/create_manifest.py
    
    # Percorsi (li adattiamo alla struttura creata prima)
    RAW_DATA_DIR = "data/raw"
    OUTPUT_MANIFEST = "data/manifests/train_manifest.json"
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"ATTENZIONE: La cartella {RAW_DATA_DIR} non esiste ancora.")
        print("Creala e mettici dentro le registrazioni divise per cartelle!")
    else:
        create_manifest(RAW_DATA_DIR, OUTPUT_MANIFEST)