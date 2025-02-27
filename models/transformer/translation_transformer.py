import torch 
import torch.nn as nn
from models.transformer.encoder import PositionalEncoder
from models.transformer import *
from models.masks import *


class TranslationTransformer(nn.Module):
    def __init__(self, de_dataset, en_dataset, d, num_heads, num_layers, ff_dim, max_len, p, device='cpu'):
        super(TranslationTransformer, self).__init__()

        self.device = torch.device(device)

        self.inp_mask = PaddingMask(en_dataset.pad_id)
        self.tgt_mask = LookAheadMask()

        self.pe = PositionalEncoder(d, max_len)
        self.embedding_encoder = nn.Embedding(de_dataset.vocab_size, d)
        self.embedding_decoder = nn.Embedding(en_dataset.vocab_size, d)

        self.encoder_layers = nn.ModuleList([
            Encoder(d, num_heads, ff_dim, p) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            Decoder(d, num_heads, ff_dim, p) for _ in range(num_layers)
        ])

        self.out = nn.Linear(d, en_dataset.vocab_size)

    def forward(self, inp, tgt):

        encoder_mask = self.inp_mask(inp)
        tgt_mask = self.tgt_mask(tgt.shape[1]).to(self.device)

        inp = self.pe(self.embedding_encoder(inp))
        tgt = self.pe(self.embedding_decoder(tgt))


        encoder_output = inp
        for l in self.encoder_layers:
            encoder_output = l(encoder_output, mask=encoder_mask)

        decoder_output = tgt
        for l in self.decoder_layers:
            decoder_output = l(decoder_output, encoder_output, tgt_mask=tgt_mask, encoder_mask=encoder_mask)

        
        output = self.out(decoder_output)

        return output