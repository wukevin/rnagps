import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np

class Basset(nn.Module):
    """
    This model is also known to do well in transcription factor binding.
    This model is "shallower" than factorized basset, but has larger convolutions
    that may be able to pick up longer motifs
    """
    def __init__(self, dropout, num_classes):
        super(Basset, self).__init__()
        torch.manual_seed(3278)
        self.dropout = dropout

        self.conv1 = nn.Conv2d(4, 300, (19, 1), stride = (1, 1), padding=(9,0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride = (1, 1), padding = (5,0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride = (1, 1), padding = (4,0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()                          # batch_size x 4 x 1000
        s = s.view(-1, 4, 1000, 1)                                   # batch_size x 4 x 1000 x 1 [4 channels]
        s = self.maxpool1(F.relu(self.bn1(self.conv1(s))))           # batch_size x 300 x 333 x 1
        s = self.maxpool2(F.relu(self.bn2(self.conv2(s))))           # batch_size x 200 x 83 x 1
        s = self.maxpool3(F.relu(self.bn3(self.conv3(s))))           # batch_size x 200 x 21 x 1
        s = s.view(-1, 4200)
        
        s = F.dropout(F.relu(self.bn4(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        s = F.dropout(F.relu(self.bn5(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        intermediate_out = s
        
        s = self.fc3(s)
        s = torch.sigmoid(s)

        return s, intermediate_out

class FactorizedBasset(nn.Module):
    """
    This model is known to do well in predicting transcription factor binding. This means it may be good
    at predicting sequence localization as well, if its architecture lends itself well to predicting sequence
    motifs in general.
    """
    def __init__(self, dropout, num_classes=1):
        super(FactorizedBasset, self).__init__()
        torch.manual_seed(3278)

        self.dropout = dropout
        self.num_cell_types = num_classes

        self.layer1 = self.layer_one()
        self.layer2 = self.layer_two()
        self.layer3 = self.layer_three()
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(4200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        # self.fc3 = nn.Linear(1000, self.num_cell_types)
        self.fc3 = nn.Linear(1000, num_classes)

    def layer_one(self):
        self.conv1a = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1b = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1c = nn.Conv2d(64, 100, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1d = nn.Conv2d(100, 150, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv1e = nn.Conv2d(150, 300, (7, 1), stride=(1, 1), padding=(3, 0))

        self.bn1a = nn.BatchNorm2d(48)
        self.bn1b = nn.BatchNorm2d(64)
        self.bn1c = nn.BatchNorm2d(100)
        self.bn1d = nn.BatchNorm2d(150)
        self.bn1e = nn.BatchNorm2d(300)

        tmp = nn.Sequential(self.conv1a, self.bn1a, nn.ReLU(inplace=True),
                            self.conv1b, self.bn1b, nn.ReLU(inplace=True),
                            self.conv1c, self.bn1c, nn.ReLU(inplace=True),
                            self.conv1d, self.bn1d, nn.ReLU(inplace=True),
                            self.conv1e, self.bn1e, nn.ReLU(inplace=True))

        return tmp

    def layer_two(self):
        self.conv2a = nn.Conv2d(300, 200, (7,1), stride = (1,1), padding = (3,0))
        self.conv2b = nn.Conv2d(200, 200, (3,1), stride = (1,1), padding = (1, 0))
        self.conv2c = nn.Conv2d(200, 200, (3, 1), stride =(1,1), padding = (1,0))

        self.bn2a = nn.BatchNorm2d(200)
        self.bn2b = nn.BatchNorm2d(200)
        self.bn2c = nn.BatchNorm2d(200)

        tmp = nn.Sequential(self.conv2a,self.bn2a, nn.ReLU(inplace= True),
                            self.conv2b,self.bn2b, nn.ReLU(inplace=True),
                            self.conv2c, self.bn2c, nn.ReLU(inplace=True))
        return tmp

    def layer_three(self):
        self.conv3 = nn.Conv2d(200, 200, (7,1), stride =(1,1), padding = (4,0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))

    def forward(self, s):
        """Expect input batch_size x 1000 x 4"""
        s = s.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
        s = s.view(-1, 4, 1000, 1)  # batch_size x 4 x 1000 x 1 [4 channels]
        s = self.maxpool1(self.layer1(s)) # batch_size x 300 x 333 x 1
        s = self.maxpool2(self.layer2(s)) # batch_size x 200 x 83 x 1
        s = self.maxpool3(self.layer3(s)) # batch_size x 200 x 21 x 1
        s = s.view(-1, 4200)
        conv_out = s
        s = F.dropout(F.relu(self.bn4(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        s = F.dropout(F.relu(self.bn5(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 1000
        s = self.fc3(s)
        s = torch.sigmoid(s)
        return s, conv_out

if __name__ == "__main__":
    # Easy sanity check that nothing is blatantly wrong
    x = FactorizedBasset(dropout=0.2, num_classes=8)
