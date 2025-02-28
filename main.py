import hydra
from omegaconf import DictConfig
from trainer.translator_trainer import TranslationTrainer
import torch
import numpy as np

@hydra.main(config_path="/Users/vsevolodparamonov/MachineTranslator/configs", config_name="default", version_base="1.1")
def main(cfg: DictConfig):

    np.random.seed(cfg.exp.seed)
    torch.random.manual_seed(cfg.exp.seed)
    
    trainer = TranslationTrainer(cfg)

    trainer.setup()
    
    if not cfg.inference.inference_mode:
        trainer.training_loop()

    trainer.translate_text(cfg.inference.test_path, 'test_pred')

if __name__ == "__main__":
    main()