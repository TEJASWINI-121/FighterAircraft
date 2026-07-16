import os
import hashlib
import cv2
from PIL import Image

def get_image_hashes(file_path):
    # Retrieve MD5 of file content and MD5 of pixel data
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Load image via cv2 to get pixel hash
        img = cv2.imread(file_path)
        if img is None:
            return file_hash, None, None
        
        pixel_hash = hashlib.md5(img.tobytes()).hexdigest()
        return file_hash, pixel_hash, img.shape
    except Exception as e:
        return None, None, None

def main():
    base_dir = r"c:\Users\sanjai\\.gemini\antigravity\scratch\FighterAircraftRecognition"
    dataset_files_dir = os.path.join(base_dir, "dataset_files")
    
    print("Auditing dataset_files subdirectories...")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_images = []
    
    for root, dirs, files in os.walk(dataset_files_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                all_images.append(full_path)
                
    print(f"Total image files found under dataset_files: {len(all_images)}")
    
    corrupt_files = []
    file_hashes = {}
    pixel_hashes = {}
    resolutions = {}
    resolution_counts = {}
    
    for path in all_images:
        f_hash, p_hash, shape = get_image_hashes(path)
        if f_hash is None or p_hash is None:
            corrupt_files.append(path)
            continue
            
        # Track file hash duplicates
        if f_hash not in file_hashes:
            file_hashes[f_hash] = []
        file_hashes[f_hash].append(path)
        
        # Track pixel hash duplicates
        if p_hash not in pixel_hashes:
            pixel_hashes[p_hash] = []
        pixel_hashes[p_hash].append(path)
        
        # Track resolutions
        resolutions[path] = shape
        resolution_counts[shape] = resolution_counts.get(shape, 0) + 1
        
    print(f"\nCorrupt files count: {len(corrupt_files)}")
    for f in corrupt_files[:10]:
        print(f"  Corrupt: {f}")
        
    file_dup_count = sum(len(paths) - 1 for paths in file_hashes.values() if len(paths) > 1)
    pixel_dup_count = sum(len(paths) - 1 for paths in pixel_hashes.values() if len(paths) > 1)
    
    print(f"\nDuplicate files (file content hash) count: {file_dup_count}")
    print(f"Duplicate images (pixel digest hash) count: {pixel_dup_count}")
    
    # Print pixel duplicate examples
    print("\nSample pixel duplicates:")
    count = 0
    for phash, paths in pixel_hashes.items():
        if len(paths) > 1:
            count += 1
            print(f"Duplicate set {count}:")
            for p in paths:
                print(f"  - {p}")
            if count >= 5:
                break
                
    # Check for extremely low resolution images
    low_res_threshold = 64 # Let's define low resolution check
    low_res_files = []
    for path, shape in resolutions.items():
        h, w, c = shape
        if h < low_res_threshold or w < low_res_threshold:
            low_res_files.append((path, shape))
            
    print(f"\nLow resolution images (H or W < {low_res_threshold}): {len(low_res_files)}")
    for path, shape in low_res_files[:10]:
        print(f"  Low res: {shape} - {path}")

if __name__ == "__main__":
    main()
