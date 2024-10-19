import torch
import torch.nn as nn
import torch.nn.functional as F


# 31% test accuracy (epochs = 15, maybe less would be better)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(CustomCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.dropout1 = nn.Dropout(p=0.25)  # Dropout after first layer
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout after second layer
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer to downsample feature map
        self.dropout3 = nn.Dropout(p=0.3)  # Dropout after third layer
        
        # Fourth Convolutional Layer (added)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer to reduce size further
        self.dropout4 = nn.Dropout(p=0.4)  # Dropout after fourth layer
        
        # Fifth Convolutional Layer (added)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.dropout5 = nn.Dropout(p=0.4)  # Dropout after fifth layer
        
        # Fully Connected Layers with gradual reduction
        self.fc1 = nn.Linear(4096, 1024)  # Adjusted to reflect smaller input size
        self.fc_dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1024, 512)  # Gradually reduce to 512
        self.fc3 = nn.Linear(512, num_classes)  # Output layer for classification

    def forward(self, x):
        # Forward pass through conv1 -> batch norm -> relu -> pool -> dropout
        x = self.pool1(self.dropout1(nn.ReLU()(self.bn1(self.conv1(x)))))
        
        # Forward pass through conv2 -> batch norm -> relu -> pool -> dropout
        x = self.pool2(self.dropout2(nn.ReLU()(self.bn2(self.conv2(x)))))
        
        # Forward pass through conv3 -> relu -> pool -> dropout
        x = self.pool3(self.dropout3(nn.ReLU()(self.conv3(x))))
        
        # Forward pass through conv4 -> relu -> pool -> dropout
        x = self.pool4(self.dropout4(nn.ReLU()(self.conv4(x))))

        # Forward pass through conv5 -> relu -> pool -> dropout
        x = self.pool5(self.dropout5(nn.ReLU()(self.conv5(x))))
        
        # Flatten the output from the convolutional layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers with gradual reduction
        x = self.fc_dropout(nn.ReLU()(self.fc1(x)))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        return x



# Test Accuracy: 35.80% (for 25 epochs)
# Test Accuracy: 31.32% (for 19 epochs)
class YOLOLikeCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(YOLOLikeCNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        
        # Fourth Convolutional Layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Fifth Convolutional Layer
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        
        # Sixth Convolutional Layer
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Seventh Convolutional Layer
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        
        # Eighth Convolutional Layer
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        
        # Ninth Convolutional Layer
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Tenth Convolutional Layer
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(1024 * 4 * 4*2, 1024)  # Flattening size will vary depending on the input size
        self.fc_dropout = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Convolutional Layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv10(x))
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x
