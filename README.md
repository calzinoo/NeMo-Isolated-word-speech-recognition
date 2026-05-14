# NeMo-Isolated-word-speech-recognition

la repository è divisa in due cartelle (scripts e game)

in scripts avviene il fine tuining del modello
- linux
- ordine: slice_audio -> augemnt_audio -> create_manifest -> train_finetune
in game sono presenti i file di gioco
- main.py per il gioco
- live_inference.py per testare il modello da solo

ogni cartella ha un file requirement.txt riservato.