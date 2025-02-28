import torch
from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence


class TranslatorDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.idx = list(range(len(self.dataset)))


        if shuffle:
            self.idx = torch.randperm(len(self.idx)).tolist()

        self.batches = [self.idx[i:i+self.batch_size] for i in range(0, len(self.idx), self.batch_size)]

    def __iter__(self):
        for batch_indices in self.batches:
            batch1, batch2 = zip(*[self.dataset[i] for i in batch_indices])
            batch1 = pad_sequence(batch1, batch_first=True, padding_value=self.dataset.dataset_de.pad_id)
            batch2 = pad_sequence(batch2, batch_first=True, padding_value=self.dataset.dataset_de.pad_id)
            yield batch1, batch2

    def __len__(self):
        return len(self.batches)