import torch.nn as nn
import torch

import sys

class Translator(nn.Module):
    def __init__(self, encoder, decoder, config, device):
        super(Translator, self).__init__()
        self.device = device
        self.config = config

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target=None):
        encoder_outputs, encoder_hidden = self.encoder(source)

        # sys.stdout.write(f'\n{encoder_outputs.shape}, {encoder_hidden.shape}\n')

        # sys.stdout.write(f'\n{self.decoder}')

        outputs, _, _ = self.decoder(
            encoder_outputs, encoder_hidden, target_tensor=target
        )

        return outputs
    
