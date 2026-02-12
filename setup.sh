#!/bin/bash

# 1. Crea l'ambiente virtuale (assicurandosi di usare Python 3.10)
echo "--- Creazione Environment con Python 3.10 ---"
python3.10 -m venv nemo_env

# 2. Attiva l'ambiente
echo "--- Attivazione Environment ---"
source nemo_env/bin/activate

# 3. Aggiorna pip
echo "--- Aggiornamento PIP ---"
pip install --upgrade pip

# 4. Installa PyTorch (Versione GPU)
echo "--- Installazione PyTorch (CUDA 11.8) ---"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Installa le altre dipendenze dal file requirements.txt
echo "--- Installazione NeMo e altre librerie ---"
# Nota: Cython va installato spesso prima di NeMo per evitare errori di compilazione
pip install cython
pip install -r requirements.txt

echo "--- TUTTO FATTO! ---"
echo "Per iniziare scrivi: source nemo_env/bin/activate"