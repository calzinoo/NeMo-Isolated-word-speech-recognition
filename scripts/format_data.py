import os
from pydub import AudioSegment

BASE_DIR = "data/raw"
TARGET_FOLDER = ["salta", "striscia", "rotola", "background"]
TARGET_LEN = 1000  # 1 secondo in millisecondi
TARGET_SR = 16000  # 16 kHz

def normalize_audio():
    print(f"--- INIZIO NORMALIZZAZIONE AUDIO ---")
    print(f"Target: Mono, {TARGET_SR} Hz, {TARGET_LEN} ms, formato WAV")

    for folder in TARGET_FOLDER:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Attenzione: cartella {folder_path} non trovata, salto...")
            continue

        print(f"Processando cartella: {folder}...")
        files = os.listdir(folder_path)
        count = 0

        for file in files:
            #accettiamo .wav, .mp3, .m4a e .ogg
            if not file.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac')):
                print(f" -> Ignorato (estensione non supportata): {file}")
                continue

            full_path = os.path.join(folder_path, file)
            try:
                #caricamento
                audio = AudioSegment.from_file(full_path)
                #conversione (mono, 16kHz)
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(TARGET_SR)
                #gestione durata
                current_length = len(audio)

                if current_length < TARGET_LEN:
                    #troppo corto, aggiungo silenzio
                    silence_needed = TARGET_LEN - current_length
                    silence = AudioSegment.silent(duration=silence_needed)
                    audio = audio + silence
                elif current_length > TARGET_LEN:
                    #troppo lungo, taglio
                    audio = audio[:TARGET_LEN]

                #esportazione in WAV
                new_filename = os.path.splitext(file)[0] + ".wav"
                new_path = os.path.join(folder_path, new_filename)

                #esporta in WAV (sovrascrivendo se già esiste)
                audio.export(new_path, format="wav")

                #pulizia
                #se il file originale non era già WAV, rimuoviamo l'originale
                if not full_path.endswith(".wav"):
                    os.remove(full_path)
                
                count += 1

            except Exception as e:
                print(f"Errore processando {file}: {e}")
            
            print(f"completati {count} file in {folder}.")

if __name__ == "__main__":
    normalize_audio()