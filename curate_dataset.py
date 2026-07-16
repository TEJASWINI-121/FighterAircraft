import os
import shutil
import random
import hashlib
import cv2
import numpy as np

# Set reproducibility seeds
random.seed(42)
np.random.seed(42)

TARGET_CLASSES = [
    "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
    "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
]

TARGET_TOTAL_IMAGES = 4800
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def normalize_name(name):
    return "".join(c for c in name.lower() if c.isalnum())

def get_dhash(img_gray, hash_size=8):
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    binary_str = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(binary_str, 2):016x}"

def hamming_distance(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count('1')

def apply_random_augmentations(img):
    # Apply rotation, flip, zoom, brightness, contrast, translation
    h, w = img.shape[:2]
    
    # 1. Horizontal Flip (50% chance)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        
    # 2. Random Rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # 3. Random Zoom (0.85 to 1.15 scale)
    scale = random.uniform(0.85, 1.15)
    nh, nw = int(h * scale), int(w * scale)
    if scale > 1.0:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dy = (nh - h) // 2
        dx = (nw - w) // 2
        img = resized[dy:dy+h, dx:dx+w]
    else:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dy = (h - nh) // 2
        dx = (w - nw) // 2
        img = cv2.copyMakeBorder(resized, dy, h - nh - dy, dx, w - nw - dx, cv2.BORDER_REFLECT_101)
        
    # 4. Random Brightness (-30 to 30 pixel shift)
    val = random.randint(-30, 30)
    img = np.clip(img.astype(np.int32) + val, 0, 255).astype(np.uint8)
    
    # 5. Random Contrast (0.85 to 1.15)
    factor = random.uniform(0.85, 1.15)
    img = np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
    
    # 6. Random Translation (up to 10% height/width)
    max_tx = int(w * 0.1)
    max_ty = int(h * 0.1)
    tx = random.randint(-max_tx, max_tx)
    ty = random.randint(-max_ty, max_ty)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    return img

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_files_dir = os.path.join(base_dir, "dataset_files")
    original_fighter_dir = os.path.join(base_dir, "fighter_dataset")
    
    norm_to_target = {normalize_name(tc): tc for tc in TARGET_CLASSES}
    
    # Count real files before expansion in the original_fighter_dir if it exists
    real_original_count = 0
    if os.path.exists(original_fighter_dir):
        for root, dirs, files in os.walk(original_fighter_dir):
            # Exclude existing augmented files in counting real files if any
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS) and not file.startswith("aug_"):
                    real_original_count += 1
                    
    # Scan all directories in dataset_files for target classes
    # We also scan fighter_dataset, but we deduplicate and clean everything
    scan_search_dirs = [dataset_files_dir]
    if os.path.exists(original_fighter_dir):
        scan_search_dirs.append(original_fighter_dir)

    print(f"Scanning target classes across directories...")
    images_by_class = {tc: [] for tc in TARGET_CLASSES}
    
    additional_collected_count = 0
    
    for search_root in scan_search_dirs:
        for root, dirs, files in os.walk(search_root):
            dir_name = os.path.basename(root)
            norm_dir = normalize_name(dir_name)
            
            if norm_dir in norm_to_target:
                target_cls = norm_to_target[norm_dir]
                for file in files:
                    if file.lower().endswith(IMAGE_EXTENSIONS) and not file.startswith("aug_"):
                        full_path = os.path.join(root, file)
                        images_by_class[target_cls].append(full_path)
                        if search_root == dataset_files_dir:
                            # It is from additional datasets
                            # Note: Veri-Kumesi is inside dataset_files, but we count it under the merge
                            additional_collected_count += 1

    print("Running quality screening and deduplication...")
    cleaned_images_by_class = {tc: [] for tc in TARGET_CLASSES}
    
    corrupt_count = 0
    dup_count = 0
    low_res_count = 0
    empty_count = 0
    
    unique_hashes = set()
    
    for tc in TARGET_CLASSES:
        for path in images_by_class[tc]:
            # 1. Empty Check
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                empty_count += 1
                continue
                
            # 2. Corrupt / Open Check
            try:
                img = cv2.imread(path)
                if img is None:
                    corrupt_count += 1
                    continue
            except Exception:
                corrupt_count += 1
                continue
                
            # 3. Resolution Check
            h, w = img.shape[:2]
            if h < 128 or w < 128:
                low_res_count += 1
                continue
                
            # 4. Duplicate Check by Perceptual Hash (dHash)
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_hash = get_dhash(gray)
            except Exception:
                corrupt_count += 1
                continue
                
            # Compare with existing unique hashes (Hamming distance <= 2)
            is_dup = False
            for u_hash in unique_hashes:
                if hamming_distance(img_hash, u_hash) <= 2:
                    is_dup = True
                    break
                    
            if is_dup:
                dup_count += 1
                continue
                
            unique_hashes.add(img_hash)
            cleaned_images_by_class[tc].append(path)
            
    # Calculate sizes
    total_real_images = sum(len(paths) for paths in cleaned_images_by_class.values())
    print(f"Total Unique Real Cleaned Images: {total_real_images}")
    
    # Splits calculations
    class_splits = {}
    for tc in TARGET_CLASSES:
        paths = cleaned_images_by_class[tc]
        n = len(paths)
        # Seed shuffle for reproducibility
        random.shuffle(paths)
        
        val_size = int(n * 0.10)
        test_size = int(n * 0.20)
        train_size = n - val_size - test_size
        
        train_paths = paths[:train_size]
        val_paths = paths[train_size:train_size + val_size]
        test_paths = paths[train_size + val_size:]
        
        class_splits[tc] = {
            "Train": train_paths,
            "Validation": val_paths,
            "Test": test_paths
        }
        
    total_val_images = sum(len(class_splits[tc]["Validation"]) for tc in TARGET_CLASSES)
    total_test_images = sum(len(class_splits[tc]["Test"]) for tc in TARGET_CLASSES)
    
    # Calculate required training target
    target_train_images = TARGET_TOTAL_IMAGES - total_val_images - total_test_images
    print(f"Target Training split size: {target_train_images}")
    
    # Distribute training targets
    train_target_per_class = {tc: target_train_images // 16 for tc in TARGET_CLASSES}
    remainder = target_train_images % 16
    for i in range(remainder):
        train_target_per_class[TARGET_CLASSES[i]] += 1
        
    # We will write new dataset to a temp folder, then swap it
    temp_fighter_dir = os.path.join(base_dir, "fighter_dataset_temp")
    if os.path.exists(temp_fighter_dir):
        shutil.rmtree(temp_fighter_dir)
        
    os.makedirs(temp_fighter_dir)
    for split in ["Train", "Validation", "Test"]:
        for tc in TARGET_CLASSES:
            os.makedirs(os.path.join(temp_fighter_dir, split, tc), exist_ok=True)
            
    report_per_class = {}
    augmented_generated_total = 0
    
    for tc in TARGET_CLASSES:
        splits = class_splits[tc]
        
        # Copy Validation and Test splits first
        for fpath in splits["Validation"]:
            shutil.copy2(fpath, os.path.join(temp_fighter_dir, "Validation", tc, os.path.basename(fpath)))
            
        for fpath in splits["Test"]:
            shutil.copy2(fpath, os.path.join(temp_fighter_dir, "Test", tc, os.path.basename(fpath)))
            
        # Copy Train real images first
        for idx, fpath in enumerate(splits["Train"]):
            # Rename target file to avoid collision if duplicate filenames exist
            filename = f"real_{idx}_{os.path.basename(fpath)}"
            shutil.copy2(fpath, os.path.join(temp_fighter_dir, "Train", tc, filename))
            
        # Perform offline augmentation on Train split if needed
        real_train_cnt = len(splits["Train"])
        target_train_cnt = train_target_per_class[tc]
        aug_needed = max(0, target_train_cnt - real_train_cnt)
        
        aug_files_cnt = 0
        if aug_needed > 0 and real_train_cnt > 0:
            for j in range(aug_needed):
                # Cycle through real training paths
                src_path = splits["Train"][j % real_train_cnt]
                img = cv2.imread(src_path)
                aug_img = apply_random_augmentations(img)
                
                # Save physically mapping
                aug_filename = f"aug_{j}_{os.path.basename(src_path)}"
                cv2.imwrite(os.path.join(temp_fighter_dir, "Train", tc, aug_filename), aug_img)
                aug_files_cnt += 1
                augmented_generated_total += 1
                
        report_per_class[tc] = {
            "Real_Train": real_train_cnt,
            "Aug_Train": aug_files_cnt,
            "Validation": len(splits["Validation"]),
            "Test": len(splits["Test"]),
            "Total": real_train_cnt + aug_files_cnt + len(splits["Validation"]) + len(splits["Test"])
        }
        
    # Swap clean directory
    if os.path.exists(original_fighter_dir):
        shutil.rmtree(original_fighter_dir)
    os.rename(temp_fighter_dir, original_fighter_dir)
    print("Dataset directory swapped successfully.")
    
    # Save Report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("          DATASET CURATION & EXPANSION PIPELINE REPORT          ")
    report_lines.append("=" * 60)
    report_lines.append(f"Number of target classes:         16")
    report_lines.append(f"Number of real images before expansion (original): {real_original_count}")
    report_lines.append(f"Images collected from additional datasets:         {additional_collected_count}")
    report_lines.append(f"Duplicate images removed (perceptual check):       {dup_count}")
    report_lines.append(f"Corrupted images removed:                          {corrupt_count}")
    report_lines.append(f"Empty files removed:                               {empty_count}")
    report_lines.append(f"Extremely low-res (< 128x128) removed:             {low_res_count}")
    report_lines.append(f"Augmented images generated (training split only):  {augmented_generated_total}")
    report_lines.append("-" * 60)
    report_lines.append("CLASS DISTRIBUTION SUMMARY:")
    report_lines.append(f"{'Class Name':<15} | {'Real Train':<10} | {'Aug Train':<10} | {'Validation':<10} | {'Test':<10} | {'Total':<10}")
    report_lines.append("-" * 60)
    for tc in TARGET_CLASSES:
        r = report_per_class[tc]
        report_lines.append(f"{tc:<15} | {r['Real_Train']:<10} | {r['Aug_Train']:<10} | {r['Validation']:<10} | {r['Test']:<10} | {r['Total']:<10}")
    report_lines.append("-" * 60)
    
    train_total = sum(report_per_class[tc]["Real_Train"] + report_per_class[tc]["Aug_Train"] for tc in TARGET_CLASSES)
    val_total = sum(report_per_class[tc]["Validation"] for tc in TARGET_CLASSES)
    test_total = sum(report_per_class[tc]["Test"] for tc in TARGET_CLASSES)
    total_size = train_total + val_total + test_total
    
    report_lines.append(f"Final Train Split size (Real + Aug): {train_total}")
    report_lines.append(f"Final Validation Split size (Real):   {val_total}")
    report_lines.append(f"Final Test Split size (Real):         {test_total}")
    report_lines.append(f"Final Total Dataset Size:             {total_size}")
    report_lines.append("=" * 60)
    
    report_txt = "\n".join(report_lines)
    
    report_path = os.path.join(base_dir, "results", "dataset_expansion_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_txt)
        
    print(f"Report saved to {report_path}")
    print(report_txt)

if __name__ == "__main__":
    main()
