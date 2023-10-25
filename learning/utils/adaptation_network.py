import torch


class AdaptationNetwork(torch.nn.Module):
    def __init__(self, input_dims, e_dims, complex=True):
        super().__init__()

        self.complex = complex
        print('Adaptation Net complex', complex)
        
        channels = 32
        if self.complex:
            channels = 64
        self.layer1 = torch.nn.Conv1d(in_channels=input_dims, out_channels=channels, kernel_size=8, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5)
        self.layer3 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=5)
        
        if self.complex:
            self.fc = torch.nn.Linear(in_features=14*channels, out_features=32)
            self.fc2 = torch.nn.Linear(in_features=32, out_features=32)
            self.fc3 = torch.nn.Linear(in_features=32, out_features=e_dims)
        else:
            self.fc = torch.nn.Linear(in_features=14*32, out_features=e_dims)
        
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act1(self.layer2(x))
        x = self.act1(self.layer3(x))
        
        x = torch.flatten(x, start_dim=1, end_dim=2)
        if self.complex:
            x = self.act1(self.fc(x))
            x = self.act1(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc(x)

        return x