import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Desining our dataaugmentations [transformations]
# transform.compose is used to chain multiple transform operations

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p = 0.5),
    #50% chance to flip the image left to right
     
    transforms.RandomRotation(degrees = 15),
    # Rotate image by +- 15 degress randomly

    transforms.ToTensor(),
    #converts the physical images to pytorch tensors (4d arrays (B,C,H,W)) 
    
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # uncomment this if using ur custom model
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    #Standardizes pixel values btw -1 and 1

])
# we dont augment the validation / test data 
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
])

# writing our custom Pytorch Dataset
class CIFAR10KaggleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        # runs only once when we initialize the data
        # Load the CSV spreadsheet into memory
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # mapping the outputs from string to numbers [encoding]
        # we have 10 classes and for classification we calculatee the probablilty of each output and for that we need out put in form of a number not string

        classes = sorted(self.labels_df['label'].unique())
        self.class_to_idx = {cls_name : i for i, cls_name in enumerate(classes)}

    def __len__(self):
        # tells pytorch how many samples are in the dataset
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # this is called 100's of times every sec 
        # this gets one image and its label during the training process 

         # 1. Get the data row for this specific index
        row = self.labels_df.iloc[idx]
        
        # 2. Reconstruct the file name (e.g., '1' -> '1.png')
        img_name = f"{row['id']}.png"
        img_path = os.path.join(self.root_dir, img_name)
        
        # 3. Open the image file
        image = Image.open(img_path).convert('RGB')
        
        # 4. Get the label string and convert it to a number
        label_str = row['label']
        label_idx = self.class_to_idx[label_str]
        
        # 5. Apply our augmentations (if any)
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

def create_dataloaders(csv_path, img_dir, batch_size=32):
    """
    Creates and returns the training DataLoader.
    (Later we will also split some data off for validation here!)
    """
    # 1. Create the Dataset object (loads 1 image at a time)
    train_dataset = CIFAR10KaggleDataset(
        csv_file=csv_path, 
        root_dir=img_dir, 
        transform=train_transform
    )
    
    # 2. Wrap it in a DataLoader (groups them into batches of 32)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle so the model doesn't memorize the order
        num_workers=2  # Uses multiple CPU cores to load images faster
    )
    
    return train_loader

# Let's test it at the bottom of the script!
if __name__ == "__main__":
    csv_path = r"C:\Users\naral\.cache\kagglehub\competitions\cifar-10\trainLabels.csv"
    img_dir = r"d:\sanket\Neural Networks\CIFAR-10\data\train"
    
    # Create the dataset object
    dataset = CIFAR10KaggleDataset(csv_file=csv_path, root_dir=img_dir, transform=train_transform)
    
    # Grab the very first item to make sure it works
    img_tensor, label = dataset[0]
    
    print(f"Success! Output tensor shape: {img_tensor.shape}")
    print(f"Output label integer: {label}")