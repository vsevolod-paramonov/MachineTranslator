import torch
import torch.nn as nn
import sys

class Decoder(nn.Module):
    def __init__(self, d, num_heads, ff_dim, p):
        super(Decoder, self).__init__()

        self.self_attention = nn.MultiheadAttention(d, num_heads, p, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, num_heads, p, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d)
            )

        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(p)

    def forward(self, inp, encoder_output, tgt_mask, encoder_mask):

        attn_output, _ = self.self_attention(inp, inp, inp, attn_mask=tgt_mask)
        attn_output = self.norm1(inp + self.dropout(attn_output))

        attn_output, _ = self.cross_attention(attn_output, encoder_output, encoder_output, key_padding_mask=encoder_mask)
        attn_output = self.norm2(inp + self.dropout(attn_output))

        ff_output = self.ff(attn_output)
        attn_output = self.norm3(attn_output + self.dropout(ff_output))

        return attn_output