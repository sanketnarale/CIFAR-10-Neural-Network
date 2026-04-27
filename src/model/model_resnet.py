import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # download the pretrained resnet 18 model 
        self.resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

        """
        one main thing about using a predefined model such as the resnet or any other model
        is that the models are already trained and output of those models are quite different 
        the final layer of ResNet18 is classifying the images into 1000 classes but
        for our dataset we only have 10 classes so we need to replace the final layer that is the fully connected layer
        and in ResNet18 the final layer is named as fc [fully connected layer]
        before we chop the fc, we need to check if there are any mathematical connects coming to it
        """

        num_features = self.resnet.fc.in_features
        print("Number of features coming to the final layer:", num_features)

        # now chop the final layer and replace it 
        # we overwrite the 1000 output layer with our blank 10 output layer

        self.resnet.fc = nn.Linear(in_features = num_features, out_features = 10)
        # the prev layer output of resnet is the input for our new layer and the output is to 10 classes
    def forward(self, x):
        # we dont need to map all the 18 layers .resnet takes care of that 
        # we just need to push the image in the model
        return self.resnet(x)

    # Test the model!
if __name__ == "__main__":
    # Create a fake batch of 5 images
    fake_images = torch.randn(5, 3, 32, 32) 
    
    # Initialize our new model
    model = CIFAR10ResNet()
    
    # Pass the fake images through
    predictions = model(fake_images)
    
    print(f"Input shape: {fake_images.shape}")
    print(f"Output shape: {predictions.shape}") # Should still be [5, 10]!

    



