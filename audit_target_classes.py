import os
import hashlib
import cv2
from PIL import Image

def get_image_hashes(file_path):
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        img = cv2.imread(file_path)
        if img is None:
            return file_hash, None, None
        
        pixel_hash = hashlib.md5(img.tobytes()).hexdigest()
        return file_hash, pixel_hash, img.shape
    except Exception as e:
        return None, None, None

def main():
    base_dir = r"c:\Users\sanjai\.gemini\antigravity\scratch\FighterAircraftRecognition"
    target_classes = [
        "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
        "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
    ]
    
    scan_paths = []
    
    # 1. existing dataset
    fighter_dir = os.path.join(base_dir, "fighter_dataset")
    if os.path.exists(fighter_dir):
        for split in ["Train", "Validation", "Test"]:
            split_dir = os.path.join(fighter_dir, split)
            if os.path.exists(split_dir):
                for tc in target_classes:
                    class_dir = os.path.join(split_dir, tc)
                    if os.path.exists(class_dir):
                        for f in os.listdir(class_dir):
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                                scan_paths.append(os.path.join(class_dir, f))
                                
    # 2. supplementary directories
    supplementary_folders = {
        "F16": "F-16",
        "F35": "F-35",
        "Mig29": "MiG-29",
        "Rafale": "Rafale"
    }
    
    dataset_files_dir = os.path.join(base_dir, "dataset_files")
    if os.path.exists(dataset_files_dir):
        for tc, folder_name in supplementary_folders.items():
            class_dir = os.path.join(dataset_files_dir, folder_name)
            if os.path.exists(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        scan_paths.append(os.path.join(class_dir, f))
                        
    out_lines = []
    out_lines.append(f"Total target class files to scan: {len(scan_paths)}")
    
    corrupt_files = []
    file_hashes = {}
    pixel_hashes = {}
    resolutions = {}
    
    for path in scan_paths:
        f_hash, p_hash, shape = get_image_hashes(path)
        if f_hash is None or p_hash is None:
            corrupt_files.append(path)
            continue
            
        if f_hash not in file_hashes:
            file_hashes[f_hash] = []
        file_hashes[f_hash].append(path)
        
        if p_hash not in pixel_hashes:
            pixel_hashes[p_hash] = []
        pixel_hashes[p_hash].append(path)
        
        resolutions[path] = shape
        
    out_lines.append(f"\nCorrupt files count: {len(corrupt_files)}")
    for f in corrupt_files:
        out_lines.append(f"  Corrupt: {f}")
        
    file_dup_count = sum(len(paths) - 1 for paths in file_hashes.values() if len(paths) > 1)
    pixel_dup_count = sum(len(paths) - 1 for paths in pixel_hashes.values() if len(paths) > 1)
    
    out_lines.append(f"\nDuplicate files (file content hash) count: {file_dup_count}")
    out_lines.append(f"Duplicate images (pixel digest hash) count: {pixel_dup_count}")
    
    out_lines.append("\nDuplicate sets (by pixel hash):")
    count = 0
    for phash, paths in pixel_hashes.items():
        if len(paths) > 1:
            count += 1
            out_lines.append(f"Duplicate set {count}:")
            for p in paths:
                out_lines.append(f"  - {p}")
            if count >= 20:
                break
                
    low_res_threshold = 64
    low_res_files = []
    for path, shape in resolutions.items():
        h, w, c = shape
        if h < low_res_threshold or w < low_res_threshold:
            low_res_files.append((path, shape))
            
    out_lines.append(f"\nLow resolution images (H or W < {low_res_threshold}): {len(low_res_files)}")
    for path, shape in low_res_files:
        out_lines.append(f"  Low res: {shape} - {path}")

    # Write output file
    out_path = os.path.join(base_dir, "audit_results.txt")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(out_lines))
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()
