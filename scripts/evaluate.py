import nemo.collections.asr as nemo_asr
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import time

# --- CONFIGURAZIONE ---
MODEL_PATH = "models/dino_finetuned.nemo"
TEST_MANIFEST = "data/manifests/train_manifest.json" # <--- CAMBIA CON test_manifest.json PER L'ESAME!
BATCH_SIZE = 16

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Valutazione su: {device}")

    # 1. Carica il Modello
    if not os.path.exists(MODEL_PATH):
        print("Errore: Modello non trovato. Hai fatto il training?")
        return
        
    model = nemo_asr.models.EncDecClassificationModel.restore_from(MODEL_PATH)
    model.eval()
    model.to(device)

    # 2. Configura il Test Data Loader
    # NeMo ha bisogno di sapere dove prendere i file di test
    model.setup_test_data(test_data_config={
        'manifest_filepath': TEST_MANIFEST,
        'sample_rate': 16000,
        'labels': model.cfg.labels,
        'batch_size': BATCH_SIZE,
        'shuffle': False
    })

    # 3. Esegui l'Inference e Misura la Latenza
    print("Inizio inferenza...")
    y_true = []
    y_pred = []
    latencies = []

    with torch.no_grad():
        for batch in model.test_dataloader():
            input_signal, input_signal_length, labels = batch
            input_signal = input_signal.to(device)
            input_signal_length = input_signal_length.to(device)
            labels = labels.to(device) # Indici numerici veri

            # Misura tempo per questo batch
            start_time = time.time()
            logits = model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
            end_time = time.time()
            
            # Calcola latenza media per singolo file (ms)
            batch_latency = (end_time - start_time) / input_signal.size(0) * 1000
            latencies.append(batch_latency)

            # Ottieni predizioni
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 4. Calcola le Metriche
    acc = accuracy_score(y_true, y_pred)
    avg_latency = np.mean(latencies)
    
    print("\n" + "="*40)
    print(f"RISULTATI VALUTAZIONE")
    print("="*40)
    print(f"Accuratezza Totale: {acc*100:.2f}%")
    print(f"Latenza Media (Inference): {avg_latency:.2f} ms")
    print("-" * 40)
    
    # Mappa indici -> nomi classi
    class_labels = model.cfg.labels
    print("\nReport Dettagliato:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # 5. Genera Matrice di Confusione (Il grafico bello)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predetto dal Modello')
    plt.ylabel('Realtà (Vero)')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    
    # Salva invece di mostrare (perché sei su WSL)
    output_img = "confusion_matrix.png"
    plt.savefig(output_img)
    print(f"\nGrafico salvato come '{output_img}'. Aprilo da Windows!")

import os
if __name__ == "__main__":
    main()