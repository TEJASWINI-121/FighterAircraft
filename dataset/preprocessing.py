import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

class FighterAircraftPreprocessor:
    """
    A class to construct high-performance TensorFlow tf.data preprocessing pipelines 
    for the fighter aircraft dataset. Coordinates image loading, resizing, conversion 
    to RGB, ResNet50 preprocessing, class mapping, caching, and prefetching.
    """
    
    def __init__(self, dataset_dir: str, batch_size: int = 32, image_size: Tuple[int, int] = (224, 224)):
        """
        Initializes the FighterAircraftPreprocessor with paths and parameters.
        
        Args:
            dataset_dir (str): Root dataset folder containing Train/, Validation/, and Test/.
            batch_size (int): Size of batches generated.
            image_size (tuple): Width and height target size (W, H).
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.splits = ["Train", "Validation", "Test"]
        self.class_names = [
            "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
            "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
        ]
        self.datasets = {}
        self.dataset_sizes = {}
        self.apply_runtime_augmentation = False  # Disabled because offline augmentation has already been applied

    def load_and_preprocess(self) -> Dict[str, tf.data.Dataset]:
        """
        Builds the tf.data input pipelines for Train, Validation, and Test data.
        Performs resizing, RGB conversion, ResNet50 preprocess centering, caching, and prefetching.
        
        Returns:
            dict: Dictionary of tf.data.Dataset indexed by splits.
        """
        print(f"[INFO] Initializing preprocessing pipelines from: {self.dataset_dir}")
        
        for split in self.splits:
            split_path = os.path.join(self.dataset_dir, split)
            if not os.path.exists(split_path):
                print(f"[WARNING] Split folder '{split}' not found at: {split_path}")
                continue
                
            # 1. Load directory using image_dataset_from_directory
            # This infers labels, decodes image files, scales aspect ratio using bilinear,
            # and enforces RGB color mode.
            ds = tf.keras.utils.image_dataset_from_directory(
                directory=split_path,
                labels="inferred",
                label_mode="int",  # Encode labels to [0-15] integer values
                class_names=self.class_names,  # Ensures class mapping is consistent
                color_mode="rgb",
                batch_size=self.batch_size,
                image_size=self.image_size,
                shuffle=(split == "Train")  # Shuffle Train split, keep Val/Test ordered
            )
            
            # Get exact file counts for logging
            num_samples = 0
            for class_folder in self.class_names:
                class_dir = os.path.join(split_path, class_folder)
                if os.path.exists(class_dir):
                    num_samples += len([
                        f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                    ])
            self.dataset_sizes[split] = num_samples

            # 2. Normalize and center pixels specifically for pretrained ResNet50 weights:
            # - Swaps RGB -> BGR
            # - Subtracts ImageNet channel-wise mean [103.939, 116.779, 123.68]
            ds = ds.map(
                lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # 3. High Performance Optimization Pipeline
            # Caches training dataset elements in-memory so files are not read repeatedly.
            # Prefetches elements to GPU/CPU while current batch is being processed.
            ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            
            self.datasets[split] = ds
            print(f"[SUCCESS] Configured split '{split}' with {num_samples} images ({len(ds)} batches).")
            
        return self.datasets

    def save_preprocessing_report(self, save_path: str) -> None:
        """
        Creates and saves a text report detailing the execution variables.
        
        Args:
            save_path (str): Filepath to save the txt report.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report = []
        report.append("=" * 60)
        report.append("          IMAGE PREPROCESSING REPORT - MODULE 2          ")
        report.append("=" * 60)
        report.append(f"Image Source Location:  {self.dataset_dir}")
        report.append(f"Target Height & Width:  {self.image_size[0]} x {self.image_size[1]} (RGB)")
        report.append(f"Normalization Mode:     ResNet50 preprocess_input (ImageNet Mean Subtraction)")
        report.append(f"ImageNet Channel Means: B: 103.939, G: 116.779, R: 123.68 (BGR format)")
        report.append(f"Pipeline Batch Size:    {self.batch_size}")
        report.append(f"Number of Classes:      {len(self.class_names)}")
        report.append("-" * 60)
        report.append("ALIGNED ENCODINGS (Alphabetical order):")
        for idx, class_name in enumerate(self.class_names):
            report.append(f"  Index {idx:<2} -> Class: {class_name}")
        report.append("-" * 60)
        report.append("DATASET PARTITION COUNTS:")
        for split in self.splits:
            cnt = self.dataset_sizes.get(split, 0)
            num_batches = len(self.datasets[split]) if split in self.datasets else 0
            report.append(f"  - {split:<12}: {cnt:<5} images ({num_batches} batches)")
        report.append("-" * 60)
        report.append("DATASET PIPELINE OPTIMIZATIONS:")
        report.append("  1. In-memory caching: Enabled (.cache())")
        report.append("  2. Parallel pre-processing elements execution: AUTOTUNE")
        report.append("  3. Overlapped training step fetching execution: AUTOTUNE")
        report.append(f"  4. Runtime Data Augmentation: Disabled (Offline augmentation applied: {not self.apply_runtime_augmentation})")
        report.append("-" * 60)
        report.append("[STATUS] READY - Preprocessing pipeline configured. Compatible with pretrained ResNet50.")
        report.append("=" * 60)
        
        report_txt = "\n".join(report)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        print(f"[INFO] Saved preprocessing report to: {save_path}")

    def display_and_save_batch(self, split: str, save_path: str) -> None:
        """
        Fetches one batch of data, displays statistics, and saves a preview grid.
        De-processes the ResNet50 channel shifts back to RGB uint8 for valid graphic plotting.
        
        Args:
            split (str): Split name to display (e.g. 'Train').
            save_path (str): Filepath to save the plot figure.
        """
        if split not in self.datasets:
            print(f"[WARNING] Dataset split '{split}' is not loaded.")
            return

        # Fetch first batch using generator
        for images, labels in self.datasets[split].take(1):
            img_array = images.numpy()
            lbl_array = labels.numpy()
            break
            
        print("\n" + "=" * 50)
        print(f"BATCH VERIFICATION DETAILS ({split} split):")
        print(f"  - Images Batch Tensor shape: {img_array.shape}")
        print(f"  - Labels Batch Tensor shape: {lbl_array.shape}")
        print(f"  - Pixel Value range:         {img_array.min():.4f} to {img_array.max():.4f}")
        print("   (Centering around zero is expected due to ImageNet mean subtraction)")
        print("=" * 50)

        # Plot first 8 images inside a 2x4 grid
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(8, len(img_array))):
            # De-process the BGR image back to RGB uint8 for visualization
            img_bgr = img_array[i].copy()
            # 1. Add back ImageNet channel means
            img_bgr[..., 0] += 103.939  # Blue
            img_bgr[..., 1] += 116.779  # Green
            img_bgr[..., 2] += 123.68   # Red
            
            # 2. Swap BGR -> RGB representation
            img_rgb = np.zeros_like(img_bgr)
            img_rgb[..., 0] = img_bgr[..., 2]
            img_rgb[..., 1] = img_bgr[..., 1]
            img_rgb[..., 2] = img_bgr[..., 0]
            
            # 3. Clip bounds to [0, 255] and map to uint8 representation
            img_disp = np.clip(img_rgb, 0, 255).astype(np.uint8)
            
            axes[i].imshow(img_disp)
            lbl_idx = lbl_array[i]
            lbl_name = self.class_names[lbl_idx]
            axes[i].set_title(f"Label: {lbl_name} ({lbl_idx})", fontsize=10, fontweight='bold')
            axes[i].axis('off')
            
        # Hide remaining grids
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f"Preprocessed Sample Batch Previews ({split} Split)", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved batch preview grid visualization to: {save_path}")
        plt.close()
