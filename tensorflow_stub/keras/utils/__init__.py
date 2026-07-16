import os
import cv2
import numpy as np
from tensorflow_stub.data import Dataset

def image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
):
    """
    Simulates tf.keras.utils.image_dataset_from_directory.
    Loads and resizes images from subdirectory structures using OpenCV.
    """
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
        
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ])
        
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    items = []
    
    # We iterate over class folders under directory
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        if not os.path.exists(class_path):
            continue
            
        try:
            files = sorted(os.listdir(class_path))
        except OSError:
            files = []
            
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(class_path, f)
                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert to RGB
                    if color_mode == "rgb":
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize to (image_size[0], image_size[1])
                    # Note: OpenCV resize takes size as (width, height), so we swap image_size (height, width)
                    img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
                    
                    items.append((img.astype(np.float32), class_to_idx[class_name]))
                    
    # Construct simulated Dataset
    ds = Dataset(items)
    ds._batch_size = batch_size
    ds._shuffle = shuffle
    return ds
