import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Function to preprocess images from a folder
def preprocess_images(folder_path, label_map, target_size=(224, 224)):
    images = []
    labels = []
    
    folder_name = os.path.basename(folder_path)
    label = label_map.get(folder_name, -1)  # Assign -1 if folder name not found in label_map
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust based on your image file types
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                img = img.resize(target_size)  # Resize image
                img = np.array(img) / 255.0    # Normalize pixel values
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    return images, labels

# Main function to preprocess all folders
def main():
    main_directory = r'C:\Users\saideepak\OneDrive\Desktop\abhi\hmm'  # Adjust this path to your main directory
    
    label_map = {
        'folder1': 0,  # Assuming folder1 corresponds to abnormal ECG images
        'folder2': 1,  # Assuming folder2 corresponds to history of MI ECG images
        'folder3': 2,  # Assuming folder3 corresponds to Myocardial Infarction Patients ECG images
        'folder4': 3   # Assuming folder4 corresponds to normal patients ECG images
    }
    
    all_images = {0: [], 1: [], 2: [], 3: []}  # Dictionary to store images by class
    all_labels = {0: [], 1: [], 2: [], 3: []}  # Dictionary to store labels by class
    
    # Process each folder in the main directory
    for folder_name in sorted(os.listdir(main_directory)):
        folder_path = os.path.join(main_directory, folder_name)
        if os.path.isdir(folder_path):
            images, labels = preprocess_images(folder_path, label_map)
            class_label = label_map.get(folder_name, -1)
            if class_label != -1:
                all_images[class_label].extend(images)
                all_labels[class_label].extend(labels)
    
    # Convert lists to numpy arrays
    for key in all_images:
        all_images[key] = np.array(all_images[key])
        all_labels[key] = np.array(all_labels[key])
    
    # Split data into training and validation sets equally across classes
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    for key in all_images:
        images = all_images[key]
        labels = all_labels[key]
        
        # Split current class data into train and validation sets
        train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(images, labels, test_size=0.2, random_state=42)
        
        # Append to the final lists
        train_images.append(train_imgs)
        train_labels.append(train_lbls)
        val_images.append(val_imgs)
        val_labels.append(val_lbls)
    
    # Concatenate arrays for final training and validation sets
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_images = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    
    # Shuffle the data
    train_shuffle_idx = np.random.permutation(len(train_images))
    val_shuffle_idx = np.random.permutation(len(val_images))
    
    train_images = train_images[train_shuffle_idx]
    train_labels = train_labels[train_shuffle_idx]
    val_images = val_images[val_shuffle_idx]
    val_labels = val_labels[val_shuffle_idx]
    
    # Print shapes to verify
    print(f"Training images shape: {train_images.shape}, Training labels shape: {train_labels.shape}")
    print(f"Validation images shape: {val_images.shape}, Validation labels shape: {val_labels.shape}")
    
    # Save preprocessed data if needed
    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    np.save('val_images.npy', val_images)
    np.save('val_labels.npy', val_labels)

if __name__ == "__main__":
    main()
