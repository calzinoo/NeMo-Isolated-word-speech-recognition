import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# CONFIGURA QUI I PERCORSI
RAW_FOLDER = "data/raw"
TARGET_LEN = 1000  # 1 secondo in millisecondi

def slice_folder(class_name):
    folder_path = os.path.join(RAW_FOLDER, class_name)
    print(f"Processando classe: {class_name}...")
    
    for file in os.listdir(folder_path):
        if not file.endswith(('.wav', '.m4a', '.mp3')): continue
        
        file_path = os.path.join(folder_path, file)
        audio = AudioSegment.from_file(file_path)
        
        # Dividi basandoti sul silenzio (dbFS è il volume)
        chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-40)
        
        for i, chunk in enumerate(chunks):
            # Normalizza durata a 1 secondo (aggiunge silenzio se serve)
            if len(chunk) < TARGET_LEN:
                silence_duration = TARGET_LEN - len(chunk)
                silence = AudioSegment.silent(duration=silence_duration)
                chunk = chunk + silence
            else:
                chunk = chunk[:TARGET_LEN] # Taglia se troppo lungo
            
            # Esporta in formato NeMo (16kHz, Mono, WAV)
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            out_name = f"{class_name}_{i}_{file}"
            if not out_name.endswith(".wav"): out_name += ".wav"
            
            chunk.export(os.path.join(folder_path, out_name), format="wav")
            print(f" -> Creato: {out_name}")
        
        # Rimuovi il file originale lungo per pulizia
        os.remove(file_path)

# Lancia per background
for label in ["salta", "striscia", "rotola"]:
    if os.path.exists(os.path.join(RAW_FOLDER, label)):
        slice_folder(label)