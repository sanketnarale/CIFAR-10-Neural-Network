import os
import torch
import pandas as pd
import py7zr
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Import our CNN Brain!
# Import our CNN Brain!
# ==============================================================================
# ⚠️ HOW TO SWAP MODELS ⚠️
# If you build a new model (e.g., EfficientNet), you must change 4 things in this file:
# 1. The import line below (Import your new model class)
# 2. The initialization (model = YourNewModel().to(device) around line 95)
# 3. The weights path (model_weights_path = ... around line 100)
# 4. The plot directory (plot_dir = ... around line 150)
# 5. The output CSV name (df.to_csv(...) around line 160)
# ==============================================================================

import sys
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/model')
# from model1 import CIFAR10CNN
from model_resnet import CIFAR10ResNet
# for plots
sys.path.append('d:/sanket/Neural Networks/CIFAR-10/src/utils')
from visualize import plot_predictions

# --- 1. PREPARE THE DATA ---
kaggle_path = r"C:\Users\naral\.cache\kagglehub\competitions\cifar-10"
local_data_dir = r"d:\sanket\Neural Networks\CIFAR-10\data"
test_zip_path = os.path.join(kaggle_path, "test.7z")
test_extract_path = os.path.join(local_data_dir, "test")

# Extract the test images. This folder has 300,000 images!
if not os.path.exists(test_extract_path):
    print("Extracting test.7z... Go grab a coffee, this will take a while!")
    with py7zr.SevenZipFile(test_zip_path, mode='r') as z:
        z.extractall(path=local_data_dir)
    print("Extraction complete!")


# --- 2. CUSTOM TEST DATASET ---
class CIFAR10TestDataset(Dataset):
    """
    This Dataset is different from the training one because we DO NOT have labels.
    Kaggle hides the labels so we can't cheat. Our job is to guess them!
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # os.listdir grabs all the file names (e.g., ['1.png', '2.png', ...])
        self.image_files = os.listdir(root_dir)
        
        # Notice we DO NOT use RandomHorizontalFlip here!
        # When taking a test, you don't want the paper spinning around.
        # We must use the EXACT same ImageNet math we used during training!
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ⚠️ HOW TO SWAP MODELS: If using your custom CNN (model1), comment out the line above and uncomment the line below!
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        # Tells PyTorch we have exactly 300,000 test images
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Grab the file name for this specific index (e.g., "1.png")
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 2. Open the physical image and convert to PyTorch Tensor
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 3. Kaggle requires us to submit the "id". 
        # Since the file is "1.png", we split it at the dot and keep the "1".
        # We convert it to an integer so we can sort them properly later.
        image_id = int(img_name.split('.')[0])
        
        # Return the math tensor, and the ID number. No label!
        return image, image_id


# --- 3. DECODING DICTIONARY ---
# In data_setup.py, we mapped classes to integers alphabetically (airplane=0, automobile=1...).
# But Kaggle wants the word "airplane", not the number 0.
# This dictionary reverses the process: you give it an integer, it gives you the string.
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}


# --- 4. THE INFERENCE SCRIPT ---
def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # Initialize a blank model (an empty brain)
    model = CIFAR10ResNet().to(device)
    
    # Load all the "memories" and math (weights) from our training session!
    # WARNING: Update this path if you saved it in a 'models/' folder!
    
    #model_weights_path = "d:/sanket/Neural Networks/CIFAR-10/cifar10_model.pth"
    
    # uncomment the above line if want to generate the csv for custom_CNN
    model_weights_path = "d:/sanket/Neural Networks/CIFAR-10/saved_models/ResNet18/cifar10_resnet_model.pth"

    model.load_state_dict(torch.load(model_weights_path, weights_only=True))
    
    # model.eval() is CRITICAL. It tells the model: "We are taking a test now. Don't learn, don't update."
    model.eval() 

    # Prepare the DataLoader
    test_dataset = CIFAR10TestDataset(root_dir=test_extract_path)
    
    # Batch size is 256. Because we aren't training, we use less memory, so we can process more images at once!
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # We will store all our guesses here
    results = []

    print(f"Predicting on {len(test_dataset)} images...")
    
    # torch.no_grad() tells PyTorch: "Don't bother doing calculus or tracking gradients."
    # This makes testing much faster and saves a massive amount of RAM.
    with torch.no_grad():
        
        # Loop through the 300,000 images in batches of 256
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device)
            
            # Step 1: Have the model guess.
            # Outputs shape is [256, 10]. (256 images, and 10 probability scores for each image)
            outputs = model(images)
            
            # Step 2: Find the highest score.
            # torch.max looks at the 10 scores and finds the highest one.
            # It returns the value, and the 'index' (which is the label 0-9). We only care about the index!
            _, predicted_indices = torch.max(outputs.data, 1)
            
            # Step 3: Loop through this specific batch of 256 to save the results
            for img_id, pred_idx in zip(image_ids, predicted_indices):
                
                # pred_idx is a tiny PyTorch Tensor. .item() turns it into a normal Python integer.
                # We pass that integer into our dictionary to get the word (e.g., 'dog')
                label_string = idx_to_class[pred_idx.item()]
                
                # Save it exactly how Kaggle asked: "id, label"
                results.append({"id": img_id.item(), "label": label_string})
                
            if (i + 1) % 100 == 0:
                print(f"Processed batch {i+1}/{len(test_loader)}")

    # --- Generate Visual Grid for Test Set ---
    plot_dir = r"d:\sanket\Neural Networks\CIFAR-10\saved_models\ResNet18\plots"
    # We pass 'predicted_indices' and set actual_labels=None because the Test set has no real labels!
    plot_predictions(images, predicted_indices, classes, plot_dir, "test_predictions.png", actual_labels=None)

    # --- 5. CREATE THE CSV FILE ---
    print("Saving submission.csv...")
    
    # Convert our list of dictionary results into a Pandas spreadsheet
    df = pd.DataFrame(results)
    
    # Kaggle is strict: the IDs MUST be in numerical order (1, 2, 3...)
    df = df.sort_values(by="id")
    
    # Save the spreadsheet to a file! index=False stops pandas from adding its own row numbers
    # We save it into the ResNet18/submission folder to keep everything hyper-organized!
    submission_dir = r"d:\sanket\Neural Networks\CIFAR-10\saved_models\ResNet18\submission"
    os.makedirs(submission_dir, exist_ok=True)
    csv_path = os.path.join(submission_dir, "submission_resnet.csv")
    
    df.to_csv(csv_path, index=False)
    
    print(f"Done! You can now upload {csv_path} to Kaggle!")

if __name__ == "__main__":
    generate_submission()
