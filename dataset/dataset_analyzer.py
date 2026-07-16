import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set

class FighterAircraftAnalyzer:
    """
    A class to perform dataset analysis and visualization for the fighter aircraft dataset.
    
    This includes scanning folders, checking dataset integrity (missing folders,
    corrupt images, duplicate filenames), generating statistics, and exporting plots
    (class distribution, dataset splits, sample images grid).
    """

    def __init__(self, dataset_dir: str, expected_classes: List[str] = None):
        """
        Initializes the FighterAircraftAnalyzer with dataset paths.
        
        Args:
            dataset_dir (str): The root directory containing 'Train', 'Validation', and 'Test'.
            expected_classes (list): The list of aircraft classes expected to be present.
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.splits = ["Train", "Validation", "Test"]
        self.expected_classes = expected_classes or [
            "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
            "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
        ]
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Scanner cache properties
        self.scanned = False
        self.class_data = {}         # key: class_name, value: dict of path pointers
        self.split_data = {}         # key: split, value: dict of class paths
        self.all_files = []          # List of all absolute image file paths
        self.stats = {}              # Processed statistics dictionary
        self.integrity_results = {}  # Integrity report dictionary

    def scan_dataset(self) -> Dict[str, any]:
        """
        Scans the dataset folder and aggregates statistics across splits and classes.
        
        Returns:
            dict: Scanned metadata and file paths grouped by splits and classes.
        """
        print(f"[INFO] Scanning dataset directory: {self.dataset_dir}")
        self.class_data = {c: {s: [] for s in self.splits} for c in self.expected_classes}
        self.split_data = {s: {c: [] for c in self.expected_classes} for s in self.splits}
        self.all_files = []

        # Iterate over Splits -> Classes -> Files
        for split in self.splits:
            split_path = os.path.join(self.dataset_dir, split)
            if not os.path.exists(split_path):
                continue
                
            for class_name in self.expected_classes:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                # Scan all valid image files
                try:
                    files = os.listdir(class_path)
                except OSError:
                    files = []
                    
                for f in files:
                    if f.lower().endswith(self.image_extensions):
                        file_path = os.path.join(class_path, f)
                        self.class_data[class_name][split].append(file_path)
                        self.split_data[split][class_name].append(file_path)
                        self.all_files.append((split, class_name, f, file_path))

        self.scanned = True
        self._calculate_statistics()
        return self.stats

    def _calculate_statistics(self) -> None:
        """
        Helper method to compile numerical statistics from scanned dataset arrays.
        """
        if not self.scanned:
            return

        total_images = len(self.all_files)
        
        # Initialize counts
        real_images = 0
        augmented_images = 0
        
        class_real_counts = {c: 0 for c in self.expected_classes}
        class_aug_counts = {c: 0 for c in self.expected_classes}
        class_total_counts = {c: 0 for c in self.expected_classes}
        
        split_real_counts = {s: 0 for s in self.splits}
        split_aug_counts = {s: 0 for s in self.splits}
        split_total_counts = {s: 0 for s in self.splits}
        
        # Matrix of counts per split and per class (real vs aug)
        split_class_matrix = {s: {c: {"real": 0, "aug": 0, "total": 0} for c in self.expected_classes} for s in self.splits}
        
        for split, class_name, filename, file_path in self.all_files:
            is_aug = filename.startswith("aug_")
            
            if is_aug:
                augmented_images += 1
                class_aug_counts[class_name] += 1
                split_aug_counts[split] += 1
                split_class_matrix[split][class_name]["aug"] += 1
            else:
                real_images += 1
                class_real_counts[class_name] += 1
                split_real_counts[split] += 1
                split_class_matrix[split][class_name]["real"] += 1
                
            class_total_counts[class_name] += 1
            split_total_counts[split] += 1
            split_class_matrix[split][class_name]["total"] += 1

        self.stats = {
            "total_classes": len(self.expected_classes),
            "classes_detected": sorted(list(self.expected_classes)),
            "total_images": total_images,
            "real_images": real_images,
            "augmented_images": augmented_images,
            "class_counts": class_total_counts,
            "class_real_counts": class_real_counts,
            "class_augmented_counts": class_aug_counts,
            "split_counts": split_total_counts,
            "split_real_counts": split_real_counts,
            "split_augmented_counts": split_aug_counts,
            "split_class_matrix": split_class_matrix
        }

    def check_integrity(self) -> Dict[str, any]:
        """
        Checks dataset integrity:
        1. Missing class or split folders
        2. Corrupted or unreadable image files (loads using OpenCV)
        3. Identical image filenames
        
        Returns:
            dict: Integrity report with warnings and error lists.
        """
        if not self.scanned:
            self.scan_dataset()

        print("[INFO] Checking dataset integrity...")
        missing_folders = []
        corrupted_images = []
        filename_tracker = {}  # filename -> list of (split, class, full_path)
        duplicates = []

        # 1. Check folder availability
        for split in self.splits:
            split_path = os.path.join(self.dataset_dir, split)
            if not os.path.exists(split_path):
                missing_folders.append(f"Split directory missing: {split}")
                continue
                
            for class_name in self.expected_classes:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    missing_folders.append(f"Class folder missing: {split}/{class_name}")

        # 2 & 3. Scan images for corruption and duplicate names
        for split, class_name, filename, full_path in self.all_files:
            # Duplicate filename tracking
            if filename not in filename_tracker:
                filename_tracker[filename] = []
            filename_tracker[filename].append((split, class_name, full_path))

            # Corrupt image checking
            try:
                # Read using cv2.imread and inspect shape
                img = cv2.imread(full_path)
                if img is None:
                    corrupted_images.append(full_path)
            except Exception as e:
                corrupted_images.append(f"{full_path} (Exception: {str(e)})")

        # Compile duplicates list
        for filename, occurrences in filename_tracker.items():
            if len(occurrences) > 1:
                duplicates.append({
                    "filename": filename,
                    "count": len(occurrences),
                    "locations": occurrences
                })

        # Save and print integrity warnings
        self.integrity_results = {
            "missing_folders": missing_folders,
            "corrupted_images": corrupted_images,
            "duplicates": duplicates
        }

        # Print warnings
        if missing_folders:
            print(f"[WARNING] Found {len(missing_folders)} missing directories:")
            for folder in missing_folders[:5]:
                print(f"  - {folder}")
        else:
            print("[SUCCESS] All split and class folders are present.")

        if corrupted_images:
            print(f"[WARNING] Found {len(corrupted_images)} corrupt / unreadable images:")
            for img in corrupted_images[:5]:
                print(f"  - {img}")
        else:
            print("[SUCCESS] All image files loaded successfully (no corruption detected).")

        if duplicates:
            print(f"[WARNING] Found {len(duplicates)} duplicate filenames across the directories:")
            for entry in duplicates[:3]:
                print(f"  - File '{entry['filename']}' occurs {entry['count']} times.")
        else:
            print("[SUCCESS] No duplicate filenames detected across the dataset.")

        return self.integrity_results

    def generate_statistics_report(self, save_path: str = None) -> str:
        """
        Constructs a text report of the dataset specifications.
        
        Args:
            save_path (str): Filepath to save the text file. Displayed to console if None.
            
        Returns:
            str: Generated report context.
        """
        if not self.scanned:
            self.scan_dataset()
        if not self.integrity_results:
            self.check_integrity()

        stats = self.stats
        report = []
        report.append("=" * 60)
        report.append("          MILITARY FIGHTER AIRCRAFT DATASET REPORT          ")
        report.append("=" * 60)
        report.append(f"Dataset Location:             {self.dataset_dir}")
        report.append(f"Total Fighter Classes:        {stats['total_classes']}")
        report.append(f"Total Images in Dataset:      {stats['total_images']}")
        report.append(f"  - Real Images:              {stats['real_images']}")
        report.append(f"  - Augmented Images:         {stats['augmented_images']}")
        report.append("-" * 60)
        
        total_real = stats['real_images']
        train_real = stats['split_real_counts'].get('Train', 0)
        val_real   = stats['split_real_counts'].get('Validation', 0)
        test_real  = stats['split_real_counts'].get('Test', 0)
        train_aug  = stats['split_augmented_counts'].get('Train', 0)
        final_train = stats['split_counts'].get('Train', 0)

        report.append("REAL DATASET SPLIT (based on real images only):")
        for split, real_cnt in [("Train", train_real), ("Validation", val_real), ("Test", test_real)]:
            pct = (real_cnt / total_real * 100) if total_real > 0 else 0
            report.append(f"  - {split:<12}: {real_cnt:<5} real images ({pct:.1f}%)")
        report.append("-" * 60)

        report.append("AUGMENTATION SUMMARY (Training Split Only):")
        report.append(f"  - Training Images Before Augmentation:  {train_real}")
        report.append(f"  - Augmented Images Generated:           {train_aug}")
        report.append(f"  - Final Training Dataset Size:          {final_train}")
        report.append("-" * 60)

        report.append("CLASS-WISE IMAGE DISTRIBUTION:")
        report.append(f"{'Class Name':<15} | {'Real Train':<10} | {'Aug Train':<10} | {'Val (Real)':<10} | {'Test (Real)':<10} | {'Total':<8}")
        report.append("-" * 60)
        for class_name in stats["classes_detected"]:
            m = stats["split_class_matrix"]
            c_rtrain = m["Train"][class_name]["real"]
            c_atrain = m["Train"][class_name]["aug"]
            c_val = m["Validation"][class_name]["real"]
            c_test = m["Test"][class_name]["real"]
            c_total = stats["class_counts"][class_name]
            report.append(f"{class_name:<15} | {c_rtrain:<10} | {c_atrain:<10} | {c_val:<10} | {c_test:<10} | {c_total:<8}")
        report.append("-" * 60)

        report.append("DATASET INTEGRITY STATUS:")
        report.append(f"  - Missing Folders:          {len(self.integrity_results['missing_folders'])}")
        report.append(f"  - Corrupt Image Files:      {len(self.integrity_results['corrupted_images'])}")
        report.append(f"  - Duplicate Filenames:      {len(self.integrity_results['duplicates'])}")
        
        # Overall readiness evaluation
        is_ready = (len(self.integrity_results['missing_folders']) == 0 and 
                    len(self.integrity_results['corrupted_images']) == 0)
        
        report.append("-" * 60)
        if is_ready:
            report.append("[STATUS] READY - Dataset integrity check passed. Ready for model training.")
        else:
            report.append("[STATUS] ATTENTION REQUIRED - Errors found. Correct issues before proceeding.")
        report.append("=" * 60)

        report_txt = "\n".join(report)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_txt)
            print(f"[INFO] Saved dataset report to: {save_path}")
            
        return report_txt

    def plot_class_distribution(self, save_path: str = None) -> None:
        """
        Generates and saves a horizontal grouped bar chart showing image counts 
        aggregated by class and split.
        """
        if not self.scanned:
            self.scan_dataset()

        # Prepare pandas DataFrame for grouped plotting
        data = []
        for class_name in self.stats["classes_detected"]:
            for split in self.splits:
                data.append({
                    "Class": class_name,
                    "Split": split,
                    "Count": len(self.class_data[class_name][split])
                })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index="Class", columns="Split", values="Count")
        pivot_df = pivot_df.reindex(columns=self.splits)  # Order splits

        # Plotting
        plt.style.use('seaborn-v0_8-pastel' if 'seaborn-v0_8-pastel' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Creating a beautiful stacked layout
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e'] # Slate Blue, Ice Blue, Soft Orange
        pivot_df.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.7)
        
        ax.set_title("Fighter Aircraft Categories Image Distribution", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Aircraft Subclass Folder", fontsize=12, labelpad=10)
        ax.set_ylabel("Quantity of Images", fontsize=12, labelpad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title="Dataset Partition", framealpha=0.9, loc='upper left')

        # Rotate x labels for clean visual spacing
        plt.xticks(rotation=45, ha='right')

        # Display raw sums above columns
        for idx, (label, row) in enumerate(pivot_df.iterrows()):
            total = row.sum()
            ax.text(idx, total + 1, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved class distribution plot to: {save_path}")
        plt.close()

    def plot_dataset_splits(self, save_path: str = None) -> None:
        """
        Plots a pie chart representing the proportions of dataset splits.
        """
        if not self.scanned:
            self.scan_dataset()

        splits = self.splits
        counts = [self.stats["split_counts"].get(s, 0) for s in splits]
        
        if sum(counts) == 0:
            print("[WARNING] Cannot generate split split plot: Dataset sizes are 0.")
            return

        plt.figure(figsize=(7, 7))
        colors = ['#3f72af', '#9db2bf', '#ffbb5c'] # Soft Dark Blue, Dusty Grey, Yellow-Orange
        explode = (0.05, 0, 0) # Explode Train to highlight size

        plt.pie(
            counts, 
            explode=explode, 
            labels=splits, 
            colors=colors,
            autopct='%1.1f%%', 
            shadow=True, 
            startangle=140,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        plt.title("Fighter Aircraft Dataset Partitions (Train / Val / Test)", fontsize=14, fontweight='bold', pad=15)
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved dataset splits pie chart to: {save_path}")
        plt.close()

    def plot_fighter_samples(self, save_path: str = None) -> None:
        """
        Plots a grid displaying one random sample image from each localized class.
        """
        if not self.scanned:
            self.scan_dataset()

        # Find one random image per class across splits
        samples = []
        for class_name in self.stats["classes_detected"]:
            # Gather all images in this class across all splits
            class_imgs = []
            for split in self.splits:
                class_imgs.extend(self.class_data[class_name][split])
            
            if class_imgs:
                # Select a random file
                chosen_img = random.choice(class_imgs)
                samples.append((class_name, chosen_img))

        if not samples:
            print("[WARNING] No image files exist in current dataset directories to plot.")
            return

        num_classes = len(samples)
        cols = 4
        rows = int(np.ceil(num_classes / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.ravel()

        for i, (label, img_path) in enumerate(samples):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"{label}\n({img.shape[1]}x{img.shape[0]})", fontsize=11, fontweight='bold')
            else:
                axes[i].text(0.5, 0.5, "Corrupt", ha='center', va='center', color='red', fontsize=12)
                axes[i].set_title(label, fontsize=11, fontweight='bold')
            axes[i].axis('off')

        # Hide any unused cell slots in the grid matrix
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle("Representative F-Series & Sukhoi Fighter Aircraft Classes", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[INFO] Saved fighter classes sample preview to: {save_path}")
        plt.close()
