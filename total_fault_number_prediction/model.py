import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictModel(torch.nn.Module):
    def __init__(self):
        super(PredictModel, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.est = nn.Linear(1, 1, bias=False)

        nn.init.constant_(self.est.weight, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.est(x)
        return x
