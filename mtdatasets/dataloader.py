from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence
import mtdatasets.dataset
import torch

class OneTextDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, cut_batch=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

        self.cut_batch = cut_batch

    @property
    def dataset(self):
        return super().dataset
        
    def __iter__(self):
        for batch in super().__iter__():
            yield batch


class TwoTextDataLoader(DataLoader):
    def __init__(self, datasets, batch_size=1, shuffle=False, cut_batch=False, seed=52):
        self.df1, self.df2 = datasets
        self.batch_size = batch_size
        self.cut_batch = cut_batch

        self.idx = list(range(len(self.df1)))
        if shuffle:
            torch.manual_seed(seed)  
            self.idx = torch.randperm(len(self.idx))

        self.batches = [self.idx[i:i+self.batch_size] for i in range(0, len(self.idx), self.batch_size)]

    @property
    def dataset(self):
        return self.df1

    def cut_tensor(self, tensor, eos_id):
        max_end = (torch.any(tensor == eos_id, dim=0) * torch.arange(tensor.shape[1])).max().item()
        if tensor.shape[1] != max_end + 1:
            return tensor[:, :max_end + 1]
        return tensor
    

    def __iter__(self):
        for batch_indices in self.batches:

            batch1 = [self.df1[i] for i in batch_indices]
            batch2 = [self.df2[i] for i in batch_indices]

            batch1 = pad_sequence(batch1, batch_first=True, padding_value=self.df1.pad_id)
            batch2 = pad_sequence(batch2, batch_first=True, padding_value=self.df2.pad_id)


            yield batch1, batch2


class TranslatorDataLoader:
    def __new__(cls, dataset, batch_size=1, shuffle=False, cut_batch=False, seed=52):
        if isinstance(dataset, mtdatasets.dataset.TranslatorDataset):
            return OneTextDataLoader(dataset, batch_size, shuffle, cut_batch)
        elif isinstance(dataset, tuple):
            return TwoTextDataLoader(dataset, batch_size, shuffle, cut_batch, seed)
        else:
            raise ValueError("Input dataset neither torch.Tensor nor Tuple!")
