"""
Code for recurrent models
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

DEVICE = utils.get_device(None)  # CPU

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

class GRULocalizationClassifier(nn.Module):
    """
    Note that this outputs LOGIT (i.e. sans activation). This has several implications
    * MUST train with a loss function that takes logits like BCEWIthLogitsLoss
    * For output prediction, must be fed into sigmoid
    """
    def __init__(self, embedding_dim, hidden_dim, gru_layers=1, vocab_size=6, num_outputs=8, device=DEVICE):
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
        return torch.zeros(self.gru_layers, 1, self.hidden_dim).to(self.device).cuda()

    def forward(self, sequence):
        self.hidden = self.init_hidden()
        embeds = self.base_embeddings(sequence)
        gru_out, self.hidden = self.gru(embeds.view(len(sequence), 1, -1), self.hidden)
        localization_space = self.hidden_to_local(gru_out.view(len(sequence), -1))
        return localization_space[-1, :]  # We only care about the last element

if __name__ == "__main__":
    GRULocalizationClassifier(32, 64, gru_layers=2)

