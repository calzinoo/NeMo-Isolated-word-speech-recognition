import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.modules import ConvASRDecoderClassification
from omegaconf import open_dict
import os

MY_LABELS = ["salta", "striscia", "rotola", "background"]
TRAIN_MANIFEST = "data/manifests/train_manifest.json"
TEST_MANIFEST = "data/manifests/train_manifest.json" 
MODEL_NAME = "commandrecognition_en_matchboxnet3x1x64_v2"

def main():
    print("1. Caricamento modello...")
    asr_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name=MODEL_NAME)

    print("   -> Applico fix per cuFFT e Device Mismatch...")
    original_forward = asr_model.preprocessor.forward
    
    def cpu_safe_forward(input_signal, length):
        asr_model.preprocessor.to('cpu')
        input_signal = input_signal.cpu()
        length = length.cpu()
        processed_signal, processed_length = original_forward(
            input_signal=input_signal, 
            length=length
        )
        device = torch.device("cuda:0")
        return processed_signal.to(device), processed_length.to(device)
    
    asr_model.preprocessor.forward = cpu_safe_forward
    print("   -> Fix applicato!")

    print(f"2. Chirurgia: Sostituzione Decoder per {len(MY_LABELS)} classi...")
    with open_dict(asr_model.cfg):
        asr_model.cfg.labels = MY_LABELS
    
    # FIX: feat_in deve essere 128 per MatchboxNet 3x1x64_v2
    asr_model.decoder = ConvASRDecoderClassification(
        feat_in=128, 
        num_classes=len(MY_LABELS),
    )
    print("   -> Decoder sostituito (feat_in=128)!")

    print("3. Setup Dati...")
    asr_model.setup_training_data(train_data_config={
        'manifest_filepath': TRAIN_MANIFEST,
        'sample_rate': 16000,
        'labels': MY_LABELS,
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
    })

    asr_model.setup_validation_data(val_data_config={
        'manifest_filepath': TEST_MANIFEST,
        'sample_rate': 16000,
        'labels': MY_LABELS,
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': False
    })

    asr_model.setup_optimization(optim_config={
        'name': 'novograd', 'lr': 0.005, 'betas': [0.95, 0.5], 'weight_decay': 0.001
    })

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        max_epochs=30, 
        enable_checkpointing=True, 
        logger=False,
        log_every_n_steps=1
    )

    print("4. AVVIO TRAINING...")
    trainer.fit(asr_model)

    print("5. Salvataggio...")
    os.makedirs("models", exist_ok=True)
    asr_model.save_to("models/dino_finetuned.nemo")
    print("FATTO! Modello salvato correttamente.")

if __name__ == '__main__':
    main()