import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictModel(torch.nn.Module):
    def __init__(self):
        super(PredictModel, self).__init__()
        hidden_units = 32
        self.input_layer = nn.Linear(1, hidden_units)
        self.hidden_layer1 = nn.Linear(hidden_units, hidden_units)
        self.hidden_layer2 = nn.Linear(hidden_units, hidden_units)
        self.hidden_layer3 = nn.Linear(hidden_units, hidden_units)
        self.output_layer = nn.Linear(hidden_units, 1)

        # He初期化（Kaiming初期化）
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.hidden_layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.hidden_layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.hidden_layer3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        # x = F.relu(self.hidden_layer1(x))
        # x = F.relu(self.hidden_layer2(x))
        # x = F.relu(self.hidden_layer3(x))
        # x = torch.exp(self.output_layer(x))
        x = F.softplus(self.output_layer(x))
        # x = torch.where(x < 0, torch.zeros_like(x), x)  # 負の値をゼロに変換

        return x