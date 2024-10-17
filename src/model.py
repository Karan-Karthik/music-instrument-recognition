import torch
import torch.nn as nn
import torchvision.models as models

class SpectrogramResNet(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super(SpectrogramResNet, self).__init__()
  
        # Load a pre-trained ResNet-18 and customize it for 1-channel spectrogram input
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 1-channel input (for spectrograms)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the fully connected layer to output the correct number of classes
        num_features = self.resnet.fc.in_features
        
        # Add dropout to prevent overfitting
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout with probability of 0.5
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize the new fc layer
        nn.init.xavier_uniform_(self.resnet.fc[1].weight)
        
    def forward(self, x):
        return self.resnet(x)
