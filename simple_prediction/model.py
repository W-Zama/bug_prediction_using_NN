import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictModel(torch.nn.Module):
    def __init__(self):
        super(PredictModel, self).__init__()
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
