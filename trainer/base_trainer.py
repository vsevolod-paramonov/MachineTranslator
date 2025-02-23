import os
import tqdm
import torch
from mtdatasets.dataset import TranslatorDataset
from mtdatasets.dataloader import TranslatorDataLoader
from abc import abstractmethod
from logger import logwriter
from utils.data_utils import _write_file
from metrics import bleu

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.exp.device 

    
    def setup(self):
        self.setup_experiment_dir()

        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()

        self.logwriter._log_custom_message('Datasets and DataLoaders are setted up!')

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.logwriter._log_custom_message('Models are setted up!')

    def setup_logger(self):

        exp_logs_dir = os.path.join(self.experiment_dir, 'logs')
        os.makedirs(exp_logs_dir, exist_ok=True)

        self.logwriter = logwriter.Logger(exp_logs_dir)

    def setup_experiment_dir(self):
        self.experiment_dir = os.path.join(os.getcwd(), self.config.exp.exp_dir, self.config.exp.exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def setup_datasets(self):

        self.train_dataset_de = TranslatorDataset(
                                    txt_path=self.config.train.train_path_de,
                                    exp_root=self.experiment_dir,
                                    mode='train',
                                    **self.config.tokenizer)
        
        self.train_dataset_en = TranslatorDataset(
                                    txt_path=self.config.train.train_path_en,
                                    exp_root=self.experiment_dir,
                                    mode='train',
                                    **self.config.tokenizer)
        
        
        self.val_dataset_de = TranslatorDataset(
                                    txt_path=self.config.val.val_path_de,
                                    exp_root=self.experiment_dir,
                                    mode='val',
                                    **self.config.tokenizer
                                    )
        
        self.val_dataset_en = TranslatorDataset(
                                    txt_path=self.config.val.val_path_en,
                                    exp_root=self.experiment_dir,
                                    mode='val',
                                    **self.config.tokenizer
                                    )
        
    def setup_dataloaders(self):
        self.train_loader = TranslatorDataLoader((self.train_dataset_de, self.train_dataset_en), batch_size=self.config.train.batch_size, shuffle=True)
        self.val_loader = TranslatorDataLoader((self.val_dataset_de, self.val_dataset_en), batch_size=self.config.train.batch_size, shuffle=False)


    def training_loop(self):
        self.to_train()

        self.logwriter._log_custom_message('Started fitting')

        for i in range(self.config.train.epoch_num):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            train_loss.update(val_loss)

            if i % self.config.train.val_step == 0 and i > 0:
                bleu_val = self.calc_metrics(self.config.val.val_path_de, self.config.val.val_path_en, 'val_pred')
                train_loss.update({'BLEU_val': {bleu_val}})

            if i % self.config.train.checkpoint_step == 0 and i > 0:
                self.save_checkpoint()

            self.logwriter._log_metrics(train_loss, i)

            raise ValueError

    @torch.inference_mode()
    def inference(self, path, name):

        self.to_eval()

        with open(path, 'r') as file:
            texts = file.readlines()

        output = []
        for txt in tqdm.tqdm(texts, desc='Inference', leave=False):
            output.append(self.seq_inference(txt))

        _write_file(output, os.path.join(self.experiment_dir, f'{name}.en'))

    def calc_metrics(self, to_translate_path, target_path, sample_name):

        self.inference(to_translate_path, sample_name)

        return bleu.bleu_score(os.path.join(self.experiment_dir, f'{sample_name}.en'), target_path)
            
    @abstractmethod
    def seq_inference(self, seq):
        pass

    @abstractmethod
    def setup_models(self):
        pass

    @abstractmethod
    def setup_optimizers(self):
        pass

    @abstractmethod
    def setup_losses(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def make_example(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @abstractmethod
    def to_train(self):
        pass

    @abstractmethod
    def to_eval(self):
        pass
