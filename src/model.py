import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class CustomResNet18(nn.Module):
    def __init__(self, num_classes=11):
        super(CustomResNet18, self).__init__()
        
        # Load pre-trained ResNet-18 model
        resnet = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 7 channels
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Keep only some layers from ResNet or reconfigure with simpler custom blocks
        self.resnet_layers = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Add custom convolutional layers to increase feature learning
        self.additional_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_additional1 = nn.BatchNorm2d(512)
        self.additional_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_additional2 = nn.BatchNorm2d(512)

        # Define custom fully connected layers similar to MultiInputCNN
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)  # Adjusted for potential output shape
        self.fc_dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc_dropout2 = nn.Dropout(p=0.35)
        self.fc3 = nn.Linear(1024, 256)
        self.fc_dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(256, num_classes)


    def forward(self, x):
        # Pass input through ResNet layers
        x = self.resnet_layers(x)
        
        # Pass through additional convolutional layers
        x = F.relu(self.bn_additional1(self.additional_conv1(x)))
        x = F.relu(self.bn_additional2(self.additional_conv2(x)))
        
        # Flatten the output
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc_dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc_dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc_dropout3(x)
        x = self.fc4(x)
        
        return x