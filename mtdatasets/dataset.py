import os
import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from utils.data_utils import _clear_dir

class LanguageDataset(Dataset):
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

        txt = [line.strip() for line in open(path, 'r', encoding='utf-8').readlines()][:100]
        lang = os.path.splitext(path)[1][1:]

        if mode=='train':

            _clear_dir(os.path.join(self.exp_root, f'{lang}_tokenizer'))

            SentencePieceTrainer.train(input=path, model_prefix=os.path.join(self.exp_root, f'{lang}_tokenizer'), minloglevel=3, **kwargs)

        tokenizer = SentencePieceProcessor(model_file=os.path.join(self.exp_root, f'{lang}_tokenizer') + '.model')

        return txt, tokenizer


    def _align_seq(self, seq):
        return [self.bos_id] + seq + [self.eos_id]

    def text2ids(self, texts):
        return [self.bos_id] + self.tokenizer.encode(texts) + [self.eos_id]

    def ids2text(self, ids):
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        ids = [id for id in ids if id not in {self.bos_id, self.eos_id, self.pad_id}]
        return self.tokenizer.decode(ids)
            
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self ,idx):

        sentence = self.indicies[idx]
        sentence_processed =  self._align_seq(sentence)

        return torch.tensor(sentence_processed)



class TranslatorDataset(Dataset):
    def __init__(self, dataset_de, dataset_en, max_pair_length=None):
        self.dataset_de = dataset_de
        self.dataset_en = dataset_en
        self.max_pair_length = max_pair_length
        self.filtered_indices = self._filter_pairs_by_length()

    def _filter_pairs_by_length(self):
        filtered_indices = []
        for i in range(len(self.dataset_de)):
            len1 = len(self.dataset_de.indicies[i])
            len2 = len(self.dataset_en.indicies[i])
            if len1 <= self.max_pair_length and len2 <= self.max_pair_length:
                filtered_indices.append(i)
        return filtered_indices

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        true_idx = self.filtered_indices[idx]
        sentence_de = self.dataset_de._align_seq(self.dataset_de.indicies[true_idx])
        sentence_en = self.dataset_en._align_seq(self.dataset_en.indicies[true_idx])
        return torch.tensor(sentence_de), torch.tensor(sentence_en)