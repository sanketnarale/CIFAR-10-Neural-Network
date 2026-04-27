import torch
import torch.nn as nn

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        #-----BLOCK 1-----
        # Input : 3 channel [cause we have RGB image]
        # output : 32 channel [we want it to learn 32 filters]
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding =1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size =  2, stride = 2) # halves the image H and W
        # output : (32, 16, 16)
        
        #-----BLOCK 2-----
        # Input : 32 channel [output channels of prev layer is the input here]
        # output : 64 channel [we want it to learn 64 filters]
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding =1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size =  2, stride = 2) # halves the image H and W
        # output : (64, 8, 8)
        
        #-----BLOCK 3-----
        # Input : 64 channel 
        # output : 128 channel [we want it to learn 128 filters]
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding =1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size =  2, stride = 2) # halves the image H and W
        # output : (128, 4, 4)

        # -----FULLY CONNECTED LAYER-----
        self.flatten = nn.Flatten()

        # The flattened size is 128 channels * 4 height * 4 width = 2048
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.relu4 = nn.ReLU()
        
        # Output is 10 because we have 10 categories (airplane, dog, etc.)
        self.fc2 = nn.Linear(in_features=512, out_features=10) 
    def forward(self, x):
        # Push the image through Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Push through Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Push through Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten and push through Linear layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x
# Test the model!
if __name__ == "__main__":
    # Create a fake random image tensor with shape [BatchSize, Channels, Height, Width]
    # Let's pretend we have a batch of 5 images
    fake_images = torch.randn(5, 3, 32, 32) 
    
    # Initialize the model
    model = CIFAR10CNN()
    
    # Pass the fake images through the model
    predictions = model(fake_images)
    
    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {predictions.shape}") # Should be [5, 10] (5 images, 10 predictions each)
        