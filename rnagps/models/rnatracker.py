import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism as described by RNATracker

    s_t = tanh(w * h_t + b)
    alpha_t = softmax s_t
    c = sum alpha_i h_i
    """
    def __init__(self, seq_len=4000, input_dims=64, attention_size=50):
        super().__init__()
        self.seq_len = seq_len
        self.context = nn.Linear(input_dims, attention_size)
        self.attn = nn.Linear(in_features=attention_size, out_features=1)

    def forward(self, x):
        # Takes, as input, output of LSTM portion of network
        # Input shape: [batch, seq_len, features]
        batch_size = x.shape[0]

        ctx_weights = self.context(x)  # [batch, seq_len, attn_size] = [128, 440, 50]
        ctx_weights = torch.tanh(ctx_weights)

        scores = self.attn(ctx_weights)  # [batch, seq_len, 1] = [128, 440, 1]
        scores = torch.squeeze(scores)

        attn_weights = torch.reshape(F.softmax(scores, -1), shape=(batch_size, self.seq_len, 1))  # [batch, seq_len]
        weighted = x * attn_weights
        retval = torch.sum(weighted, dim=1)  # [batch, features] = [batch, 64]
        return retval

class RNATracker(nn.Module):
    """
    Reimplementation of RNATracker

    Important details:
    - N is set to [0.25, 0.25, 0.25, 0.25] instead of all [0, 0, 0, 0]
    - Set at 4000 nt max length
    - Each CNN is 32 conv filters of length 10, xavier uniform init
    - Each pooling layer takes window of size 3 and stride of 3
    - 100 epochs of training
    - Batch size of 256
    - RNATracker uses softmax as final activation, here we might change that b/c multiclass
    """
    def __init__(self, n_classes=8, nb_filters=32, filters_length=10, pooling_size=3, lstm_units=32, attention_size=50, max_len=4000, final_act=torch.sigmoid):
        super().__init__()
        torch.manual_seed(4837)
        self.final_activation = final_act

        self.cnn1 = nn.Conv1d(
            in_channels=4,
            out_channels=nb_filters,
            kernel_size=filters_length,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.cnn1.weight)
        self.maxpool1 = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
        self.dropout1 = nn.Dropout(p=0.2)

        self.cnn2 = nn.Conv1d(
            in_channels=nb_filters,
            out_channels=nb_filters,
            kernel_size=filters_length,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.cnn2.weight)
        self.maxpool2 = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
        self.dropout2 = nn.Dropout(p=0.2)

        # LSTM expects as input (seq_len, batch, input_size) by default
        self.lstm = nn.LSTM(
            input_size=nb_filters,
            hidden_size=lstm_units,
            dropout=0.1,
            bidirectional=True,
        )

        # multiply LSTM units x2 because we don't separate bidirectional
        self.attention = Attention(seq_len=440, input_dims=lstm_units*2, attention_size=attention_size)

        self.fc = nn.Linear(lstm_units * 2, n_classes)

    def forward(self, x):
        # [batch, 4, 4000]
        assert x.shape[-1] == 4000
        batch_size = x.shape[0]
        cnn_out1 = self.dropout1(self.maxpool1(F.relu(self.cnn1(x))))  # [batch, 32, 1330]
        cnn_out2 = self.dropout2(self.maxpool2(F.relu(self.cnn2(cnn_out1))))  # [batch, 32, 440]

        # TODO masking appears to only be a computational advantage, not for correctness
        # cnn_mask = cnn_out2 == 0.  # TODO figure out columnwise check
        # cnn_masked = torch.masked_select(cnn_out2, mask=cnn_mask)
        # lstm = self.lstm(cnn_masked)

        # Permute to (seq_len, batch, input_size)
        cnn_out2_permuted = cnn_out2.permute(2, 0, 1)  # [440, batch, 32]
        lstm_out, (lstm_h_n, lstm_c_n) = self.lstm(cnn_out2_permuted)
        # lstm_out: (440, batch, 64) - does not need unpacking
        # lstm_h_n: (2, 128, 32) - hidden state
        # lstm_c_n: (2, 128, 32)
        # batch = 128, seq_len = 440, hidden x2 = 64
        lstm_out_permuted = lstm_out.permute(1, 0, 2)  # [batch, seq_len, lstm_hidden x 2]
        attn = self.attention(lstm_out_permuted)

        retval = self.final_activation(self.fc(attn))  # [batch, num_classes]
        return retval

if __name__ == "__main__":
    RNATracker()

