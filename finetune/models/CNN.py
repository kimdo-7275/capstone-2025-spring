import torch.nn as nn

class Simple_1DCNN(nn.Module):
    def __init__(self, num_leads= 12, num_classes= 5, h1= 64, h2= 128, h3= 256, kernel_size= 8):
        super(CNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(
                in_channels= num_leads,
                out_channels= h1,
                kernel_size= kernel_size
            ),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels= h1,
                out_channels= h2,
                kernel_size= kernel_size
            ),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels= h2,
                out_channels= h3,
                kernel_size= kernel_size
            ),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(h3, num_classes)
    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x).squeeze()
        x = self.dropout(x)
        x = self.fc(x)
        return x