import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, dataset, hidden_size, num_layers, dropout_p, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.de = dataset

        self.embedding = nn.Embedding(dataset.vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_size * 2, hidden_size) if bidirectional else nn.Identity()

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)

        if isinstance(self.fc, nn.Linear):
            output = self.fc(output)

        return output, hidden
