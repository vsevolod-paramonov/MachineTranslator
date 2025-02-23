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
        self.dataset = dataset
        self.embedding = nn.Embedding(dataset.vocab_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size, num_layers * (2*(multiply==True) + 1*(multiply==False)))
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=num_layers * (2*(multiply==True) + 1*(multiply==False)), batch_first=True)
        self.out = nn.Linear(hidden_size, dataset.vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.dataset.bos_id)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        assert target_tensor is not None, 'Please, insert target_tensor for faster inference'

        for _ in range(target_tensor.shape[1]):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
    
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach() 

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        input = input.to(self.device)

        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2).reshape(hidden.shape[1], 1, -1)

        context, attn_weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedded, context), dim=2)


        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights