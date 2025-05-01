import torch
import torch.nn as nn
import torch.nn.functional as F

class ShogiCNN(nn.Module):
    def __init__(self):
        super(ShogiCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=40, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 9 * 9, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 6561)  # 81×81マス

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))    # (64, 9, 9)
        x = F.relu(self.bn2(self.conv2(x)))    # (128, 9, 9)
        x = F.relu(self.bn3(self.conv3(x)))    # (256, 9, 9)
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # → (batch_size, 6561)
