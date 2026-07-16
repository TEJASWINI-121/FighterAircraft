"""
Removes near-duplicate images from all Train class folders using perceptual hashing.
Replaces removed images with fresh augmented images to maintain the training set size.
"""
import os
import cv2
import random
import numpy as np

random.seed(99)
np.random.seed(99)

TARGET_CLASSES = [
    "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
    "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def get_dhash(img_gray, hash_size=8):
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    binary_str = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(binary_str, 2):016x}"

def hamming_distance(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count('1')

def apply_random_augmentation(img):
    h, w = img.shape[:2]
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    val = random.randint(-30, 30)
    img = np.clip(img.astype(np.int32) + val, 0, 255).astype(np.uint8)
    factor = random.uniform(0.85, 1.15)
    img = np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
    return img

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fighter_dir = os.path.join(base_dir, "fighter_dataset")
    total_removed = 0

    for tc in TARGET_CLASSES:
        train_dir = os.path.join(fighter_dir, "Train", tc)
        if not os.path.exists(train_dir):
            continue

        # Gather all files, real ones first, then aug
        all_files = [f for f in os.listdir(train_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
        real_files = sorted([f for f in all_files if not f.startswith("aug_")])
        aug_files  = sorted([f for f in all_files if f.startswith("aug_")])
        ordered_files = real_files + aug_files

        seen_hashes = set()
        to_remove = []

        for fname in ordered_files:
            fpath = os.path.join(train_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                to_remove.append(fpath)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h = get_dhash(gray)
            is_dup = any(hamming_distance(h, uh) <= 1 for uh in seen_hashes)
            if is_dup:
                to_remove.append(fpath)
            else:
                seen_hashes.add(h)

        if to_remove:
            print(f"[{tc}] Removing {len(to_remove)} near-duplicate(s): {[os.path.basename(p) for p in to_remove]}")
            for path in to_remove:
                os.remove(path)
                total_removed += 1

            # Regenerate replacement augmented images using remaining real images in Training
            remaining_real = [f for f in os.listdir(train_dir) if not f.startswith("aug_") and f.lower().endswith(IMAGE_EXTENSIONS)]
            if not remaining_real:
                print(f"[{tc}] WARNING: No real images left to augment from.")
                continue
            for i, _ in enumerate(to_remove):
                src_fname = remaining_real[i % len(remaining_real)]
                src_path = os.path.join(train_dir, src_fname)
                img = cv2.imread(src_path)
                aug_img = apply_random_augmentation(img)
                aug_name = f"aug_repl_{i}_{src_fname}"
                cv2.imwrite(os.path.join(train_dir, aug_name), aug_img)
                print(f"[{tc}] Replacement augmented image saved: {aug_name}")

    if total_removed == 0:
        print("No near-duplicate images found in any Train folder.")
    else:
        print(f"\nTotal near-duplicates removed and replaced: {total_removed}")

if __name__ == "__main__":
    main()
