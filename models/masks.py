import torch

class PaddingMask:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, seq):        
        return (seq == self.pad_token_id).float().to(device=seq.device)

class LookAheadMask:
    def __init__(self):
        pass

    def __call__(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask