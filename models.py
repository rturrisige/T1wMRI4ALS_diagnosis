import torch
import torch.nn


class DNN_3HL(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=250),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=250, out_features=100),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(in_features=100, out_features=2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.softmax(self.classifier(x))
        return x


class DNN_2HL(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=300),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=300, out_features=100),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(in_features=100, out_features=2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.softmax(self.classifier(x))
        return x
