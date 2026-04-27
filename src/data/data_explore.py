import pandas as pd
import py7zr
import os
import matplotlib.pyplot as plt 

# defining the paths 
kaggle_path = r"C:\Users\naral\.cache\kagglehub\competitions\cifar-10"
local_data_dir = r"d:\sanket\Neural Networks\CIFAR-10\data"

# Extract the data but make sure its done only once not repeated again and again

train_zip_path = os.path.join(kaggle_path, "train.7z")
train_extract_path = os.path.join(local_data_dir, "train")

# check if extraction is already done if so skip 

if not os.path.exists(train_extract_path):
    print("Extracting the train.7r..... might take a min")
    os.makedirs(local_data_dir, exist_ok=True)

    # perform the extraction
    with py7zr.SevenZipFile(train_zip_path, mode='r') as z:
        z.extractall(path=local_data_dir)
    print("Extraction complete!")
else:
    print("Data already extracted!")

# Explore the dataset

# 4. Read the CSV Labels
csv_path = os.path.join(kaggle_path, "trainLabels.csv")
labels_df = pd.read_csv(csv_path)

# 5. Explore the Physical Images
image_files = os.listdir(train_extract_path)
print("\n--- Image Files Overview ---")
print(f"Total files in train folder: {len(image_files)}")
print(f"First 5 files: {image_files[:5]}") 
# This will show you the exact file extension (like .png or .jpg)

# 6. Let's inspect ONE single image deeply
from PIL import Image
# Pick the image named '1.png'
first_image_path = os.path.join(train_extract_path, "9000.png")
img = Image.open(first_image_path)
print("\n--- Deep Dive into 1.png ---")
print(f"Format: {img.format}")       # What type of file is it?
print(f"Color Mode: {img.mode}")     # Is it RGB? Grayscale (L)? RGBA?
print(f"Dimensions: {img.size}")     # Width x Height

# 7. Link the Image to the Label
# Let's find the label for '1.png' inside our CSV file
# The 'id' in the CSV is just the number '1'
label_row = labels_df[labels_df['id'] == 9000]
label_name = label_row['label'].values[0]
print(f"\nThe label for 9000.png from our CSV is: {label_name}")

# Let's visually prove it by plotting it!
plt.imshow(img)
plt.title(f"File: 9000.png | True Label: {label_name}")
plt.axis('off')
plt.show()

import torchvision.transforms as transforms

# 8. Let's see what ToTensor actually does!
to_tensor = transforms.ToTensor()

# Pass our physical image ('img') through the transform
img_tensor = to_tensor(img)

print("\n--- Tensor Transformation ---")
print(f"Shape of Tensor: {img_tensor.shape}")  # Notice it's now [Channels, Height, Width]
print(f"Data type: {img_tensor.dtype}")        # It converted it to a Float32 decimal

# Let's look at the actual numbers for the top-left corner pixel across the 3 RGB channels
print("\nTop-Left Pixel Values (Notice they are between 0.0 and 1.0!):")
print(f"Red value:   {img_tensor[0, 0, 0]}")
print(f"Green value: {img_tensor[1, 0, 0]}")
print(f"Blue value:  {img_tensor[2, 0, 0]}")

# If you want to see the entire 32x32 grid of numbers for the Red channel, 
# uncomment the line below:
print(img_tensor[0])
