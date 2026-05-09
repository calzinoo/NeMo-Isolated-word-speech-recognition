import os
from pydub import AudioSegment
import config

def detect_leading_silence(sound, silence_threshold = -40.0, chunk_size = 10):
    #trova la durata del silenzio all'inizio dell'audio
    trim_ms = 0  # ms
    assert chunk_size > 0 #evita loop infinito
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def normalize_audio():
    print(f"--- INIZIO NORMALIZZAZIONE AUDIO ---")
    print(f"Target: Mono, {config.SAMPLE_RATE} Hz, {config.TARGET_LEN_MS} ms, formato WAV")

    for folder in config.MY_LABELS:
        folder_path = os.path.join(config.RAW_DATA_DIR, folder)
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

                start_trim = detect_leading_silence(audio)
                #se c'è silenzio  taglialo
                if( start_trim > 0):
                    audio = audio[start_trim:]

                #conversione (mono, 16kHz)
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(config.SAMPLE_RATE)
                #gestione durata
                current_length = len(audio)

                if current_length < config.TARGET_LEN_MS:
                    #troppo corto, aggiungo silenzio
                    silence_needed = config.TARGET_LEN_MS - current_length
                    silence = AudioSegment.silent(duration=silence_needed)
                    audio = audio + silence
                elif current_length > config.TARGET_LEN_MS:
                    #troppo lungo, taglio
                    audio = audio[:config.TARGET_LEN_MS]

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