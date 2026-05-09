import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.modules import ConvASRDecoderClassification
from omegaconf import open_dict
import config
import os


def main():
    print("1. Caricamento modello...")
    asr_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name=config.PRETRAINED_MODEL)

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

    print(f"2. Chirurgia: Sostituzione Decoder per {len(config.MY_LABELS)} classi...")
    with open_dict(asr_model.cfg):
        asr_model.cfg.labels = config.MY_LABELS
    
    # FIX: feat_in deve essere 128 per MatchboxNet 3x1x64_v2
    asr_model.decoder = ConvASRDecoderClassification(
        feat_in=128, 
        num_classes=len(config.MY_LABELS),
    )
    print("   -> Decoder sostituito (feat_in=128)!")

    print("3. Setup Dati...")
    asr_model.setup_training_data(train_data_config={
        'manifest_filepath': config.TRAIN_MANIFEST,
        'sample_rate': config.SAMPLE_RATE,
        'labels': config.MY_LABELS,
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
    })

    asr_model.setup_validation_data(val_data_config={
        'manifest_filepath': config.TEST_MANIFEST,
        'sample_rate': config.SAMPLE_RATE,
        'labels': config.MY_LABELS,
        'batch_size': config.BATCH_SIZE,
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
        max_epochs=config.MAX_EPOCHS, 
        enable_checkpointing=True, 
        logger=False,
        log_every_n_steps=1
    )

    print("4. AVVIO TRAINING...")
    trainer.fit(asr_model)

    print("5. Salvataggio...")
    os.makedirs("models", exist_ok=True)
    asr_model.save_to(config.FINAL_MODEL_PATH)
    print("FATTO! Modello salvato correttamente.")

if __name__ == '__main__':
    main()