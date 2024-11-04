import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNNLSTM(nn.Module):
    def __init__(self, num_classes=11, input_channels=8, lstm_hidden_size=128):
        super(CustomCNNLSTM, self).__init__()

        # CNN Part with 10 convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
        )

        # Calculate the flattened dimension after the CNN layers
        # Adjust based on pooling layers; here, we assume three pooling layers (factor of 8 reduction)
        self.flattened_dim = 256 * (130 // 8) * (259 // 8)  # Adjust based on actual pooling and output size

        # LSTM Part
        # !!! Reshape input size !!!
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.dropout_fc1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN Forward Pass
        x = self.conv_layers(x)

        # Flatten and Reshape for LSTM Input
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 256)  # !!! Reshape this !!!

        # LSTM Forward Pass
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]  # Get last time step output
        x = self.lstm_dropout(x)

        # Fully Connected Forward Pass
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        return x
