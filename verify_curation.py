import os
import cv2
import numpy as np

TARGET_CLASSES = [
    "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
    "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
]

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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fighter_dir = os.path.join(base_dir, "fighter_dataset")
    
    print("====================================")
    print("      VERIFYING CURATION RESULTS    ")
    print("====================================\n")
    
    # 1. Check directories structure
    assert os.path.exists(fighter_dir), "fighter_dataset folder does not exist"
    
    splits = ["Train", "Validation", "Test"]
    for split in splits:
        split_path = os.path.join(fighter_dir, split)
        assert os.path.exists(split_path), f"Split folder {split} missing"
        
        folders = sorted(os.listdir(split_path))
        # Check that there are only target classes in folders
        for folder in folders:
            assert folder in TARGET_CLASSES, f"Unexpected folder {folder} in split {split}"
        for tc in TARGET_CLASSES:
            assert tc in folders, f"Target class folder {tc} missing in split {split}"
            
    print("[PASS] Check 1: Directory structures and class directories are correct.")
    
    # 2. Check no data leakage (Validation/Test should contain NO augmented files)
    augmented_files_in_val = []
    augmented_files_in_test = []
    
    for tc in TARGET_CLASSES:
        val_dir = os.path.join(fighter_dir, "Validation", tc)
        for f in os.listdir(val_dir):
            if f.startswith("aug_"):
                augmented_files_in_val.append(os.path.join(val_dir, f))
                
        test_dir = os.path.join(fighter_dir, "Test", tc)
        for f in os.listdir(test_dir):
            if f.startswith("aug_"):
                augmented_files_in_test.append(os.path.join(test_dir, f))
                
    assert len(augmented_files_in_val) == 0, f"Found augmented images in Validation: {augmented_files_in_val}"
    assert len(augmented_files_in_test) == 0, f"Found augmented images in Test: {augmented_files_in_test}"
    
    print("[PASS] Check 2: Validation and Test splits contain 0 augmented images (no data leakage).")
    
    # 3. Check dataset size range (total dataset size must be within 4,500 - 5,000)
    total_images_cnt = 0
    all_final_paths = []
    
    for split in splits:
        for tc in TARGET_CLASSES:
            class_dir = os.path.join(fighter_dir, split, tc)
            for f in os.listdir(class_dir):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    total_images_cnt += 1
                    all_final_paths.append(os.path.join(class_dir, f))
                    
    print(f"Total physical images in final dataset: {total_images_cnt}")
    assert 4500 <= total_images_cnt <= 5000, f"Total images size {total_images_cnt} is outside 4500-5000 range"
    print("[PASS] Check 3: Total dataset size matches target bounds (4,500-5,000 images).")
    
    # 4. Check image integrity & resolution threshold
    print("Verifying image channels, resolutions, and corruption status...")
    min_resolution_pass = True
    corruption_pass = True
    
    unique_hashes = set()
    dup_detected = 0
    
    for path in all_final_paths:
        # File integrity
        img = cv2.imread(path)
        if img is None:
            corrupt_pass = False
            print(f"[FAIL] Corrupted file: {path}")
            continue
            
        # Resolution threshold
        h, w = img.shape[:2]
        if h < 128 or w < 128:
            min_resolution_pass = False
            print(f"[FAIL] Low resolution ({w}x{h}): {path}")
            
        # Deduplication check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hash = get_dhash(gray)
        is_dup = False
        for u_hash in unique_hashes:
            if hamming_distance(img_hash, u_hash) <= 1: # Strict check
                is_dup = True
                break
        if is_dup:
            dup_detected += 1
            print(f"[WARNING] Similar image detected: {path}")
        else:
            unique_hashes.add(img_hash)
            
    assert corruption_pass, "Found corrupted images in the final dataset"
    assert min_resolution_pass, "Found images below 128x128 resolution in the final dataset"
    assert dup_detected == 0, f"Found {dup_detected} near-duplicate image(s) after deduplication. All duplicates must be removed before completion."
    
    print("[PASS] Check 4: All images decode correctly (no corruptions detected).")
    print("[PASS] Check 5: All image dimensions are >= 128x128 pixels.")
    print(f"[PASS] Check 6: Zero duplicate images remain after perceptual deduplication.")
    
    # Print splits summary
    print("\nSplit Allocations Summary:")
    for split in splits:
        split_cnt = 0
        for tc in TARGET_CLASSES:
            class_dir = os.path.join(fighter_dir, split, tc)
            split_cnt += len([f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTENSIONS)])
        pct = (split_cnt / total_images_cnt) * 100
        print(f"  - {split:<12}: {split_cnt:<5} images ({pct:.1f}%)")
        
    print("\n====================================")
    print("      ALL VERIFICATION CHECKS PASS! ")
    print("====================================")

if __name__ == "__main__":
    main()
