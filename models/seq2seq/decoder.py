import torch.nn as nn
import torch.nn.functional as F
import torch


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size * num_layers, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, dataset, hidden_size, num_layers, multiply, dropout_p, device=None):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.en = dataset

        self.embedding = nn.Embedding(dataset.vocab_size, hidden_size)
        num_layers_adj = num_layers * (2 if multiply else 1)

        self.attention = BahdanauAttention(hidden_size, num_layers_adj)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=num_layers_adj, batch_first=True)

        self.out = nn.Linear(hidden_size, dataset.vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, decoder_input):
        decoder_hidden = encoder_hidden

        decoder_outputs, attentions = [], []

        for i in range(decoder_input.shape[1]):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input[:, i:i + 1], decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)

        return F.log_softmax(decoder_outputs, dim=-1), decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2).reshape(hidden.shape[1], 1, -1)
        context, attn_weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
