import torch 
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d, max_len):
        super(PositionalEncoder, self).__init__()

        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        c = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * c)
        pe[:, 1::2] = torch.cos(pos * c)

        self.register_buffer('pe', pe)

    def forward(self, inp):
        return inp + self.pe[:inp.shape[1]].to(inp.device).unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, d, num_heads, ff_dim, p):
        super(Encoder, self).__init__()

        self.attention = nn.MultiheadAttention(d, num_heads, p, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d))

        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)

    def forward(self, inp, mask=None):
        attn, _ = self.attention(inp, inp, inp, key_padding_mask=mask)

        inp = inp + self.dropout(attn)
        inp = self.norm1(inp)

        ff_res = self.ff(inp)
        inp = inp + self.dropout(ff_res)
        inp = self.norm2(inp)

        return inp
        