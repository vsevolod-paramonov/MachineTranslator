import os
import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from utils.data_utils import _clear_dir

class TranslatorDataset(Dataset):
    def __init__(self, txt_path, exp_root=None, max_length=None, mode='train', **kwargs):

        self.max_length = max_length 
        self.exp_root = exp_root
        
        self.text, self.tokenizer = self._preproc_text(txt_path, mode, **kwargs)
        self.indicies = self.tokenizer.encode(self.text)

        self.vocab_size = self.tokenizer.vocab_size()

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = self.tokenizer.pad_id(), \
                                                             self.tokenizer.unk_id(), \
                                                             self.tokenizer.bos_id(), \
                                                             self.tokenizer.eos_id()

    
    def _preproc_text(self, path, mode, **kwargs):

        txt = open(path).readlines()[:10000]
        lang = os.path.splitext(path)[1][1:]

        if mode=='train':

            _clear_dir(os.path.join(self.exp_root, f'{lang}_tokenizer'))

            SentencePieceTrainer.train(input=path, model_prefix=os.path.join(self.exp_root, f'{lang}_tokenizer'), minloglevel=3, **kwargs)

        tokenizer = SentencePieceProcessor(model_file=os.path.join(self.exp_root, f'{lang}_tokenizer') + '.model')

        return txt, tokenizer


    def _align_seq(self, seq):

        if self.max_length is not None:
            init_len = len(seq)
            
            if init_len > self.max_length - 2:
                init_len = self.max_length - 2
                seq = seq[:self.max_length-2]

            seq = [self.bos_id] + seq + [self.eos_id]
            seq += [self.pad_id] * (self.max_length - len(seq))

            return seq

        return [self.bos_id] + seq + [self.eos_id]

    def text2ids(self, texts):
        return self.tokenizer.encode(texts)

    def ids2text(self, ids):
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.tokenizer.decode(ids)
            
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self ,idx):

        sentence = self.indicies[idx]
        sentence_processed =  self._align_seq(sentence)

        return torch.tensor(sentence_processed)