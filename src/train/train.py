import torch
import torch.nn as nn
import torch.optim as optim

# Import our custom files! 
# (You might need to adjust the import depending on your exact folder structure)
import sys
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/data')
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/model')

from data_setup import create_dataloaders
from model1 import CIFAR10CNN

def train_model():
    # 1. Setup the Device (Use GPU if you have one, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Get our DataLoader
    csv_path = r"C:\Users\naral\.cache\kagglehub\competitions\cifar-10\trainLabels.csv"
    img_dir = r"d:\sanket\Neural Networks\CIFAR-10\data\train"
    
    # We will feed the network 64 images at a time
    print("Loading data...")
    train_loader = create_dataloaders(csv_path, img_dir, batch_size=64)

    # 3. Initialize Model, Loss Function, and Optimizer
    print("Initializing model...")
    model = CIFAR10CNN().to(device) # Move the model to our device
    
    # CrossEntropyLoss is the standard for Classification (it handles the 0-9 integers perfectly)
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a very smart, self-adjusting optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Epoch Loop
    epochs = 5 # Let's just do 5 rounds through the entire dataset to start
    print("Starting training!")
    
    for epoch in range(epochs):
        model.train() # Put the model in training mode
        running_loss = 0.0
        
        # Loop through batches of 64 images
        for i, (images, labels) in enumerate(train_loader):
            # Send the data to the GPU/CPU
            images = images.to(device)
            labels = labels.to(device)
            
            # --- THE 5 STEPS OF TRAINING ---
            optimizer.zero_grad()               # Step 1: Clear old math
            outputs = model(images)             # Step 2: Guess
            loss = criterion(outputs, labels)   # Step 3: Grade the guess
            loss.backward()                     # Step 4: Find the mistakes
            optimizer.step()                    # Step 5: Fix the mistakes
            
            # Keep track of our progress
            running_loss += loss.item()
            
            # Print an update every 100 batches so we know it hasn't crashed
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
                
    print("Finished Training!")
    
    # 5. Save the trained model to the hard drive!
    torch.save(model.state_dict(), "cifar10_model.pth")
    print("Model saved to cifar10_model.pth")

if __name__ == "__main__":
    train_model()
