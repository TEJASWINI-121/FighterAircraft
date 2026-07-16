import os
import shutil

def main():
    # Source path definition
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.join(base_dir, "dataset_files", "Veri-Kumesi_V1-100")
    dst_root = os.path.join(base_dir, "fighter_dataset")

    if not os.path.exists(src_root):
        print(f"[ERROR] Source dataset not found at: {src_root}")
        return

    # Define the 16 existing fighter aircraft class folders that match the user request
    fighter_folders = [
        "AV8B",       # Harrier
        "EF2000",     # Eurofighter Typhoon
        "F14",        # F-14 Tomcat
        "F15",        # F-15 Eagle
        "F16",        # F-16 Fighting Falcon
        "F18",        # F/A-18 Hornet
        "F35",        # F-35 Lightning II
        "F4",         # F-4 Phantom II
        "J10",        # Chengdu J-10
        "J20",        # J-20 Mighty Dragon
        "JF17",       # JF-17 Thunder
        "KAAN",       # TAI TF-X / KAAN
        "Mig29",      # MiG-29
        "Mirage2000", # Mirage 2000
        "Rafale",     # Rafale
        "Su57"        # Su-57
    ]

    splits = ["Train", "Test", "Validation"]
    
    # Initialize statistics counters
    total_images_copied = 0
    class_image_counts = {c: 0 for c in fighter_folders}
    copied_classes = set()

    print("[INFO] Starting copy of fighter aircraft folders...")

    # Copy files split by split
    for split in splits:
        src_split_dir = os.path.join(src_root, split)
        dst_split_dir = os.path.join(dst_root, split)
        
        if not os.path.exists(src_split_dir):
            print(f"[WARNING] Source split directory '{split}' not found at: {src_split_dir}")
            continue
            
        os.makedirs(dst_split_dir, exist_ok=True)
        
        for folder in fighter_folders:
            src_class_path = os.path.join(src_split_dir, folder)
            dst_class_path = os.path.join(dst_split_dir, folder)
            
            if os.path.exists(src_class_path):
                copied_classes.add(folder)
                os.makedirs(dst_class_path, exist_ok=True)
                
                # Copy all images in the folder
                files = os.listdir(src_class_path)
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
                img_files = [f for f in files if f.lower().endswith(image_extensions)]
                
                for f in img_files:
                    src_file = os.path.join(src_class_path, f)
                    dst_file = os.path.join(dst_class_path, f)
                    shutil.copy2(src_file, dst_file)
                    class_image_counts[folder] += 1
                    total_images_copied += 1
            else:
                print(f"[WARNING] Fighter folder '{folder}' not found in split '{split}'")

    print("\n" + "=" * 50)
    print("           COPY COMPLETED SUCCESSFULLY           ")
    print("=" * 50)
    print(f"Total Fighter Classes Copied: {len(copied_classes)}")
    print(f"Total Images Copied:          {total_images_copied}")
    print("-" * 50)
    print(f"{'Class Folder':<15} | {'Images Copied':<12}")
    print("-" * 50)
    for folder in sorted(copied_classes):
        print(f"{folder:<15} | {class_image_counts[folder]:<12}")
    print("=" * 50)

if __name__ == "__main__":
    main()
