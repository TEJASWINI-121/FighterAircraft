import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class FighterAircraftDataset:
    """
    A class to load, inspect, and analyze the Fighter Aircraft Dataset.
    This coordinates directory structures, class discovery, image verification,
    and reporting of dataset distribution statistics.
    """
    
    def __init__(self, dataset_dir: str, expected_classes: List[str] = None):
        """
        Initializes the FighterAircraftDataset manager.
        
        Args:
            dataset_dir (str): The root directory containing aircraft class subfolders.
            expected_classes (list): List of class names to enforce/create.
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.expected_classes = expected_classes or [
            "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
            "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
        ]
        self.classes = []
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Verify and initialize directories
        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """
        Private helper to check structural integrity of directories.
        Creates missing subfolders for expected subclasses if they do not exist.
        """
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"[INFO] Created root dataset directory: {self.dataset_dir}")
            
        for class_name in self.expected_classes:
            class_path = os.path.join(self.dataset_dir, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                print(f"[INFO] Created folder for class '{class_name}': {class_path}")
                
        # Scan what folders actually exist (in alphabetical order)
        self.classes = sorted([
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ])

    def get_statistics(self) -> Dict[str, any]:
        """
        Scans the dataset directories and calculates distribution statistics.
        
        Returns:
            dict: Directory stats (class list, file counts, image dimensions).
        """
        stats = {
            "root_directory": self.dataset_dir,
            "classes": self.classes,
            "num_classes": len(self.classes),
            "counts_per_class": {},
            "total_images": 0,
            "avg_resolutions": {}
        }
        
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_dir, class_name)
            # Filter files by supported image extensions
            files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(self.image_extensions)
            ]
            count = len(files)
            stats["counts_per_class"][class_name] = count
            stats["total_images"] += count
            
            # Extract resolution info for up to 5 images to save performance
            resolutions = []
            sample_files = files[:5]
            for file_name in sample_files:
                file_path = os.path.join(class_path, file_name)
                img = cv2.imread(file_path)
                if img is not None:
                    h, w, _ = img.shape
                    resolutions.append((w, h))
                    
            if resolutions:
                avg_w = int(np.mean([r[0] for r in resolutions]))
                avg_h = int(np.mean([r[1] for r in resolutions]))
                stats["avg_resolutions"][class_name] = (avg_w, avg_h)
            else:
                stats["avg_resolutions"][class_name] = (0, 0)
                
        return stats

    def display_statistics(self) -> None:
        """
        Formats and prints dataset statistics block to the stdout.
        """
        stats = self.get_statistics()
        print("=" * 60)
        print("          FIGHTER AIRCRAFT DATASET STATISTICS          ")
        print("=" * 60)
        print(f"Dataset Root Dir:  {stats['root_directory']}")
        print(f"Total Classes:     {stats['num_classes']} {stats['classes']}")
        print(f"Total Image Files: {stats['total_images']}")
        print("-" * 60)
        print(f"{'Class Name':<15} | {'Image Count':<12} | {'Avg Resolution (W x H)':<25}")
        print("-" * 60)
        for class_name in stats["classes"]:
            count = stats["counts_per_class"][class_name]
            res = stats["avg_resolutions"][class_name]
            res_str = f"{res[0]}x{res[1]}" if res[0] > 0 else "N/A (No images)"
            print(f"{class_name:<15} | {count:<12} | {res_str:<25}")
        print("=" * 60)

    def load_samples(self, n_samples_per_class: int = 1) -> List[Tuple[str, str]]:
        """
        Retrieves path pointers to sample images for each class.
        
        Args:
            n_samples_per_class (int): Number of sample images to fetch per class.
            
        Returns:
            list: List of Tuples containing (class_name, image_path)
        """
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_dir, class_name)
            files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(self.image_extensions)
            ]
            
            # Select up to requested samples
            selected_files = files[:n_samples_per_class]
            for file_name in selected_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((class_name, file_path))
                
        return samples

    def plot_samples(self, n_samples: int = 1, save_path: str = None) -> None:
        """
        Plots representative sample images in a neat grid.
        
        Args:
            n_samples (int): Samples per class to render.
            save_path (str): Filepath to save the plot visualization, if provided.
        """
        samples = self.load_samples(n_samples)
        if not samples:
            print("[WARNING] Cannot plot: No image files exist in current dataset directories.")
            return
            
        num_pics = len(samples)
        cols = min(num_pics, len(self.classes))
        rows = int(np.ceil(num_pics / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        # Ensure axes is always a flat list
        if num_pics == 1:
            axes = [axes]
        else:
            axes = np.array(axes).ravel()
            
        for i, (label, img_path) in enumerate(samples):
            # Read with cv2 (BGR format) and swap channels to RGB for matplotlib
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"Class: {label}\nShape: {img.shape[1]}x{img.shape[0]}", fontsize=10)
            else:
                # Text fallback if corrupt file
                axes[i].text(0.5, 0.5, "Corrupt Image", ha='center', va='center')
                axes[i].set_title(label)
            axes[i].axis('off')
            
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        if save_path:
            # Ensure output folder exists
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"[INFO] Saved dataset sample visualization to: {save_path}")
        plt.show()
