"""
Code for recurrent models
"""
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNBase, LSTMCell

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

DEVICE = 'cpu'  # CPU

class mLSTM(RNNBase):
    """
    Multiplicative LSTM
    https://florianwilhelm.info/2018/08/multiplicative_LSTM_for_sequence_based_recos/
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(mLSTM, self).__init__(
            mode='LSTM', input_size=input_size, hidden_size=hidden_size,
                 num_layers=1, bias=bias, batch_first=True,
                 dropout=0, bidirectional=False)

        w_im = torch.Tensor(hidden_size, input_size)
        w_hm = torch.Tensor(hidden_size, hidden_size)
        b_im = torch.Tensor(hidden_size)
        b_hm = torch.Tensor(hidden_size)
        self.w_im = nn.Parameter(w_im)
        self.b_im = nn.Parameter(b_im)
        self.w_hm = nn.Parameter(w_hm)
        self.b_hm = nn.Parameter(b_hm)

        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        n_batch, n_seq, n_feat = input.size()

        hx, cx = hx
        steps = [cx.unsqueeze(1)]
        for seq in range(n_seq):
            mx = F.linear(input[:, seq, :], self.w_im, self.b_im) * F.linear(hx, self.w_hm, self.b_hm)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(input[:, seq, :], hx)
            steps.append(cx.unsqueeze(1))

        return torch.cat(steps, dim=1)

class LSTMLocalizationClassifier(nn.Module):
    """
    Reference:
    https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

    Note that this outputs LOGIT (i.e. sans activation). This has several implications
    * MUST train with a loss function that takes logits like BCEWIthLogitsLoss
    * For output prediction, must be fed into sigmoid
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size=6, num_outputs=8, device=DEVICE):
        # Needs vocab of 6 to account for N bases, and for 0 as a padding index
        super(LSTMLocalizationClassifier, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)  # Embedding is a lookup table - 0 index is fine
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_local = nn.Linear(hidden_dim, num_outputs)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(self.device).cuda(),
                torch.zeros(1, 1, self.hidden_dim).to(self.device).cuda())

    def forward(self, sequence):
        self.hidden = self.init_hidden()
        embeds = self.base_embeddings(sequence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sequence), 1, -1), self.hidden)
        localization_space = self.hidden_to_local(lstm_out.view(len(sequence), -1))
        return localization_space[-1, :]

class mLSTMLocalizationClassifier(nn.Module):
    """
    Multiplicative lstm classifier
    """
    def __init__(self, embedding_dim:int=32, hidden_dim:int=64, vocab_size=6, num_outputs=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)  # Embedding is a lookup table - 0 index is fine
        self.lstm = mLSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden_to_local = nn.Linear(hidden_dim, num_outputs, bias=True)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.hidden_dim).to(self.device).cuda(),
                torch.zeros(1, self.hidden_dim).to(self.device).cuda())

    def forward(self, sequence):
        l = sequence.shape[0]
        embeds = self.base_embeddings(sequence)  # (len, 1, 1, embedding_dim)
        # mlstm expects (batch, seq_len, features)
        embeds = embeds.view(l, 1, -1).permute(1, 0, 2)
        lstm_out = self.lstm(embeds, self.init_hidden())  # (batch, seq_len, hidden)
        x = lstm_out.squeeze()
        local_space = self.hidden_to_local(x[-1, :])
        return local_space

class GRULocalizationClassifier(nn.Module):
    """
    Note that this outputs LOGIT (i.e. sans activation). This has several implications
    * MUST train with a loss function that takes logits like BCEWIthLogitsLoss
    * For output prediction, must be fed into sigmoid
    """
    def __init__(self, embedding_dim:int=32, hidden_dim:int=64, gru_layers:int=2, vocab_size:int=6, num_outputs:int=8, device=DEVICE):
        super(GRULocalizationClassifier, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=gru_layers, bias=True, batch_first=False)
        self.hidden_to_local = nn.Linear(hidden_dim, num_outputs)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(self.gru_layers, 1, self.hidden_dim).to(self.device)#.cuda()

    def forward(self, sequence):
        self.hidden = self.init_hidden()
        embeds = self.base_embeddings(sequence)
        gru_out, self.hidden = self.gru(embeds.view(len(sequence), 1, -1), self.hidden)
        localization_space = self.hidden_to_local(gru_out.view(len(sequence), -1))
        return localization_space[-1, :]  # We only care about the last element

class DeepGRULocalizationClassifier(nn.Module):
    """
    Note that this outputs LOGIT (i.e. sans activation). This has several implications
    * MUST train with a loss function that takes logits like BCEWIthLogitsLoss
    * For output prediction, must be fed into sigmoid
    """
    def __init__(self, embedding_dim:int=32, hidden_dim:int=64, gru_layers:int=2, vocab_size:int=6, num_outputs:int=8, device=DEVICE):
        super(DeepGRULocalizationClassifier, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.base_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=gru_layers, bias=True, batch_first=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, num_outputs, bias=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(self.gru_layers, 1, self.hidden_dim).to(self.device)#.cuda()

    def forward(self, sequence):
        self.hidden = self.init_hidden()
        embeds = self.base_embeddings(sequence)
        gru_out, self.hidden = self.gru(embeds.view(len(sequence), 1, -1), self.hidden)
        x = F.relu(self.fc1(gru_out.view(len(sequence), -1)))
        retval = self.fc2(x)
        return retval[-1, :]  # We only care about the last element

if __name__ == "__main__":
    GRULocalizationClassifier(32, 64, gru_layers=2)

