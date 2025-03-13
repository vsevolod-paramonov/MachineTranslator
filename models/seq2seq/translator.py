import torch.nn as nn
import torch

class Translator(nn.Module):
    def __init__(self, encoder, decoder, config, device):
        super(Translator, self).__init__()
        self.device = device
        self.config = config

        self.encoder = encoder
        self.decoder = decoder

        self.de = encoder.de
        self.en = decoder.en

    def forward(self, source, target=None):
        encoder_outputs, encoder_hidden = self.encoder(source)

        if target is not None:
            outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target)
        else:
            outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, torch.full((source.size(0), 1), self.de.bos_id, dtype=torch.long, device=self.device))

        return outputs

    def encode(self, input):
        return self.encoder(input)

    def decode(self, encoder_outputs, encoder_hidden, decoder_input):
        return self.decoder(encoder_outputs, encoder_hidden, decoder_input)
