import torch
import os
import torch.nn as nn
import torch.optim as optim

# Import our custom files! 
# (You might need to adjust the import depending on your exact folder structure)
import sys
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/data')
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/model')

# ==============================================================================
# ⚠️ HOW TO SWAP MODELS ⚠️
# If you build a new model (e.g., EfficientNet), you must change 4 things in this file:
# 1. The import line below (Import your new model class)
# 2. The initialization (model = YourNewModel().to(device) around line 95)
# 3. The weights path (model_weights_path = ... around line 100)
# 4. The plot directory (plot_dir = ... around line 150)
# 5. The output CSV name (df.to_csv(...) around line 160)
# ==============================================================================

# import utils folder for plots
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/utils')
from visualize import plot_training_metrics, plot_predictions

from data_setup import create_dataloaders

# from model1 import CIFAR10CNN
from model_resnet import CIFAR10ResNet

def train_model():
    # --- 1. SETUP FOLDERS ---
    save_dir = r"d:\sanket\Neural Networks\CIFAR-10\saved_models\ResNet18"
    plot_dir = os.path.join(save_dir, "plots")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
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
    #model = CIFAR10CNN().to(device) # Move the model to our device
    model = CIFAR10ResNet().to(device)
    
    # CrossEntropyLoss is the standard for Classification (it handles the 0-9 integers perfectly)
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a very smart, self-adjusting optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dictionary to keep track of our plot data!
    history = {'loss': [], 'accuracy': []}

    # 4. The Epoch Loop
    epochs = 15 # Let's just do 15 rounds through the entire dataset
    print("Starting training!")
    
    for epoch in range(epochs):
        model.train() # Put the model in training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
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
            
            # Track Accuracy for plotting later!
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Print an update every 100 batches so we know it hasn't crashed
            if (i + 1) % 100 == 0:
                current_acc = 100 * correct_predictions / total_predictions
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}, Accuracy: {current_acc:.2f}%")
                running_loss = 0.0
                
        # Save the final accuracy/loss for this entire epoch into our history dictionary
        epoch_accuracy = 100 * correct_predictions / total_predictions
        history['accuracy'].append(epoch_accuracy)
        history['loss'].append(running_loss / len(train_loader)) # Average loss for the epoch
                
    print("Finished Training!")
    
    # 5. Save the trained model to the hard drive!
    torch.save(model.state_dict(), os.path.join(save_dir, "cifar10_resnet_model.pth"))
    
    # Save the metrics history as a JSON file (optional, but good practice!)
    import json
    with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
        json.dump(history, f)
        
    print(f"Model saved to {save_dir}")
    
    # --- 6. GENERATE PLOTS ---
    print("Generating plots...")
    plot_training_metrics(history, plot_dir)
    
    # Let's plot the very last batch of training images as a visual test!
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_predictions(images, predicted, classes, plot_dir, "training_predictions.png", actual_labels=labels)
    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    train_model()
