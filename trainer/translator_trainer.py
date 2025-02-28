from trainer.base_trainer import BaseTrainer
from hydra.utils import instantiate
import torch.nn as nn
import torch
from models.seq2seq.translator import Translator
import tqdm
import sys


class TranslationTrainer(BaseTrainer):

    def setup_models(self):
        self.device = torch.device(self.config.exp.device)         


        if 'TranslationTransformer' in self.config.translator.transformer._target_:
            self.model = instantiate(self.config.translator.transformer, de_dataset=self.de_train_text, en_dataset=self.en_train_text, 
                                      device=self.device).to(self.device)

        else:
            self.encoder = instantiate(self.config.translator.encoder, dataset=self.train_dataset_de)
            self.decoder = instantiate(self.config.translator.decoder, dataset=self.train_dataset_en, multiply=self.encoder.gru.bidirectional, device=self.device)

            self.model = Translator(encoder=self.encoder, decoder=self.decoder, config=self.config, device=self.device).to(self.device)

        if self.config.exp.checkpont_path:
            self.checkpoint = torch.load(self.config.exp.checkpont_path, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['translator_state_dict'])

        self.generator = instantiate(self.config.translator.generator, device=self.device, model=self.model)
    
    def setup_optimizers(self):
        self.optimizer = instantiate(self.config.optimizer, params=self.model.parameters())

        if self.config.exp.checkpont_path:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def setup_schedulers(self):
        self.scheduler = instantiate(self.config.scheduler, optimizer=self.optimizer)

        if self.config.exp.checkpont_path:
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
    
    def setup_losses(self):
        self.loss = nn.CrossEntropyLoss(ignore_index=self.config.tokenizer.pad_id, label_smoothing=0.1).to(self.device)


    def train_epoch(self):

        self.to_train()

        train_loss = 0.0
        
        for de, en in tqdm.tqdm(self.train_loader, desc='Train', leave=False):
            de, en = de.to(self.device), en.to(self.device)
            
            self.optimizer.zero_grad()

            pred = self.model(de, en[:, :-1])

            loss = self.loss(pred.reshape(-1, pred.size(-1)), en[:, 1:].reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.translator.grad_clip)
            self.optimizer.step()

            train_loss += loss.item() * de.shape[0]

        self.scheduler.step()


        train_loss /= len(self.train_loader.dataset)



        return {'loss_train': train_loss}
            

    def val_epoch(self):

        self.to_eval()

        val_loss = 0.0

        with torch.no_grad():
            for de, en in tqdm.tqdm(self.val_loader, desc='Validation', leave=False):
                de, en = de.to(self.device), en.to(self.device)
                    
                pred = self.model(de, en[:, :-1])[:, :en.shape[1], :]
    
                loss = self.loss(pred.reshape(-1, pred.size(-1)), en[:, 1:].reshape(-1))
    
                val_loss += loss.item() * de.shape[0]

        val_loss /= len(self.val_loader.dataset)

        return {'loss_val': val_loss}

    @torch.inference_mode()
    def make_example(self):
       return self.config.example.input, self.model.inference(self.config.example.input, self.train_dataset_de, self.train_dataset_en), self.config.example.right_output
        
    @torch.inference_mode()
    def inference(self, seq):
        return self.generator.generate(seq)

    def save_checkpoint(self):
        torch.save({
                    'translator_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                   },
                   f'{self.experiment_dir}/{self.config.exp.exp_name}_checkpoint.pth')

    def to_train(self):
        self.model.train()


    def to_eval(self):
        self.model.eval()
