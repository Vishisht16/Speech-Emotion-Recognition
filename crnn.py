import torch
import torch.nn as nn

class EmotionCRNN(nn.Module):
    def __init__(self, num_classes=8, n_mfcc=40):
        super(EmotionCRNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_mfcc, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.rnn = nn.GRU(input_size=512, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1) 
        out, _ = self.rnn(x)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out