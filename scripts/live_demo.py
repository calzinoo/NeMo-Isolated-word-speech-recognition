import pyaudio
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import os

# Configurazione PyAudio
MODEL_PATH = "models/dino_finetuned.nemo"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
BUFFER_SECONDS = 1.5
THRESHOLD = 0.70

def main():
    #caricamento del modello
    if not os.path.exists(MODEL_PATH):
        print(f"Errore: modello non trovato in {MODEL_PATH}")
        return
    
    print(f"Caricamento del modello da {MODEL_PATH}...")
    asr_model = nemo_asr.models.EncDecClassificationModel.restore_from(MODEL_PATH)

    #mettiamo il modello in modalità valutazione
    asr_model.eval()

    #applichiamo fix CPU per compatibilità driver
    print("Applicazione del fix CPU per compatibilità driver...")
    original_forward = asr_model.preprocessor.forward

    def cpu_safe_forward(input_signal, length):
        #sposta preprocessor e input su CPU
        asr_model.preprocessor.to('cpu')
        input_signal = input_signal,
        length = length,
        
        #calcola
        processed_signal, processed_length = original_forward(
            input_signal=input_signal,
            length=length
        )

        #riporta preprocessor su GPU (o CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return processed_signal.to(device), processed_length.to(device)
    
    asr_model.preprocessor.forward = cpu_safe_forward
    #spostiamo il resto del modello su GPU se disponibile
    if torch.cuda.is_available():
        print("Spostamento del modello su GPU...")
        asr_model.to('cuda:0')
    
    #setup microfono
    p = pyaudio.PyAudio()

    #calcoliamo quanto buffer serve (es. 1.5 secondi * 16000 campioni/secondo = 24000 campioni)
    buffer_len = int(SAMPLE_RATE * BUFFER_SECONDS)
    audio_buffer = np.zeros(buffer_len, dtype=np.float32)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    
    print("\n" + "="*40)
    print(f"IN ASCOLTO")
    print(f"comandi noti: {asr_model.cfg.labels}")
    print("premi Ctrl+C per terminare")
    print("="*40 + "\n")

    try:
        while True:
            #leggi i dati dal microfono
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            #converti in numpy array
            audio_data = np.frombuffer(data, dtype=np.float32)

            #aggiorna il buffer (shift a sinistra e aggiungi nuovi dati alla fine)
            audio_buffer = np.roll(audio_buffer, -len(audio_data))
            audio_buffer[-len(audio_data):] = audio_data

            #prepara il tensore per il modello (1, 1, buffer_len)
            audio_tensor = torch.tensor(audio_buffer).unsqueeze(0)
            length_tensor = torch.tensor([buffer_len])

            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
                length_tensor = length_tensor.cuda()

            #fai la previsione (senza calcolare gradiente per velocità)
            with torch.no_grad():
                logits = asr_model.forward(input_signal=audio_tensor, input_signal_length=length_tensor)
                probs = torch.softmax(logits, dim=-1)
                
                #prendiamo la classe con percentuale più alta
                prob_max, class_idx = torch.max(probs, dim=-1)

                confidence = prob_max.item()
                predicted_label = asr_model.cfg.labels[class_idx.item()]

                #stampa solo se la confidenza supera la soglia
                if confidence > THRESHOLD:
                    print(f"Rilevato: {predicted_label} (confidenza: {confidence:.2f})")

    except KeyboardInterrupt:
        print("\nTerminazione in corso...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Microfono chiuso. Arrivederci!")

if __name__ == "__main__":
    main()