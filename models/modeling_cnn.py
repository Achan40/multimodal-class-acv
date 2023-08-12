import torch
import torch.nn as nn

# Define hyperparameters for the CNN model
hidden_units = 128
dropout_rate = 0.5

# Define the CNN architecture
class MultiClassCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.batchnorm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(50176, hidden_units)
        self.batchnorm_dense = nn.BatchNorm1d(hidden_units)
        self.dense2 = nn.Linear(hidden_units, 14)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm_dense(x)
        x = self.dense2(x)
        
        return torch.sigmoid(x)
