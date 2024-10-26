import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(MultiInputCNN, self).__init__()

        # Convolutional Block for all features
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(3, 3), padding=1),  # 7 input channels for 7 features
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Reduce width gradually
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))   # Reduce width gradually
        )

        # Fully connected layers
        self.fc1 = nn.Linear(66560, 1024)  # Adjust dimensions based on conv output
        self.fc_dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, input_tensor):
        # Pass the combined feature tensor through the convolutional block
        x = self.conv_block(input_tensor)
        
        # Flatten the processed feature map
        x = torch.flatten(x, start_dim=1)

        # Pass the flattened features through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc_dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc_dropout2(x)
        x = self.fc3(x)

        return x
