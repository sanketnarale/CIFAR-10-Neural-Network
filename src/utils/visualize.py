import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(history, save_dir):
    """Plots the Loss and Accuracy line graphs."""
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy', color='green', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss', color='red', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_path)
    print(f"Metrics plot saved to {plot_path}")
    plt.close()

def plot_predictions(images, predicted_labels, classes, save_dir, filename, actual_labels=None):
    """Plots a grid of images with their predicted labels."""
    fig = plt.figure(figsize=(10, 10))
    
    # Un-normalize the images so they look normal (undoing the mean/std math)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # We will plot the first 9 images in the batch
    for i in range(9):
        plt.subplot(3, 3, i+1)
        
        # Un-normalize and convert tensor to image format [H, W, C]
        img = images[i].cpu().numpy()
        img = (img * std) + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        
        plt.imshow(img)
        
        # Get the string names
        pred_name = classes[predicted_labels[i].item()]
        
        if actual_labels is not None:
            actual_name = classes[actual_labels[i].item()]
            color = 'green' if pred_name == actual_name else 'red'
            plt.title(f"Pred: {pred_name}\nAct: {actual_name}", color=color)
        else:
            plt.title(f"Pred: {pred_name}")
            
        plt.axis('off')
        
    plt.tight_layout()
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    print(f"Prediction grid saved to {plot_path}")
    plt.close()
