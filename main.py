import os
from dataset.dataset_analyzer import FighterAircraftAnalyzer
from dataset.preprocessing import FighterAircraftPreprocessor
from model.resnet50_model import build_resnet50_model

def main():
    """
    Main entry point for driving Project Modules:
    - Module 1: Dataset Analysis & Visualization (integrity verification, stats, layout charts)
    - Module 2: Image Preprocessing Pipeline (TensorFlow tf.data, scaling, batching, caching, prefetching)
    - Module 3: Deep Learning Model Development (ResNet50 Transfer Learning architecture)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "fighter_dataset")
    results_dir = os.path.join(base_dir, "results")
    
    expected_classes = [
        "AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4",
        "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"
    ]

    print("\n" + "=" * 60)
    print("      PROJECT ENGINE: FIGHTER AIRCRAFT RECOGNITION      ")
    print("=" * 60)

    # ------------------------------------------------------------
    # MODULE 1: Dataset Analysis & Visualization
    # ------------------------------------------------------------
    print("\n>>> INITIATING MODULE 1: DATASET ANALYSIS & VISUALIZATION <<<")
    analyzer = FighterAircraftAnalyzer(dataset_dir=dataset_dir, expected_classes=expected_classes)
    
    # 1. Scan and print basic stats
    analyzer.scan_dataset()
    analyzer.check_integrity()
    
    # 2. Save and display reports
    report_save_path = os.path.join(results_dir, "dataset_statistics.txt")
    report_txt = analyzer.generate_statistics_report(save_path=report_save_path)
    
    # 3. Create plots
    dist_plot_path = os.path.join(results_dir, "class_distribution.png")
    split_plot_path = os.path.join(results_dir, "dataset_split.png")
    samples_plot_path = os.path.join(results_dir, "fighter_samples.png")
    
    analyzer.plot_class_distribution(save_path=dist_plot_path)
    analyzer.plot_dataset_splits(save_path=split_plot_path)
    analyzer.plot_fighter_samples(save_path=samples_plot_path)
    
    print("[SUCCESS] Module 1 executed, report and visual plots exported to results/.")

    # ------------------------------------------------------------
    # MODULE 2: Image Preprocessing Pipeline
    # ------------------------------------------------------------
    print("\n>>> INITIATING MODULE 2: IMAGE PREPROCESSING PIPELINE <<<")
    preprocessor = FighterAircraftPreprocessor(
        dataset_dir=dataset_dir, 
        batch_size=32, 
        image_size=(224, 224)
    )
    
    # 1. Load splits and construct pipelines
    datasets = preprocessor.load_and_preprocess()
    
    # 2. Save preprocessing execution details
    preproc_report_path = os.path.join(results_dir, "preprocessing_report.txt")
    preprocessor.save_preprocessing_report(save_path=preproc_report_path)
    
    # 3. Fetch one batch to verify shape/normalization and output a batch grid preview
    batch_preview_path = os.path.join(results_dir, "preprocessed_batch_preview.png")
    preprocessor.display_and_save_batch(split="Train", save_path=batch_preview_path)
    
    print("[SUCCESS] Module 2 executed, reports and batch plots saved in results/.")

    # ------------------------------------------------------------
    # MODULE 3: Deep Learning Model Development
    # ------------------------------------------------------------
    print("\n>>> INITIATING MODULE 3: DEEP LEARNING MODEL DEVELOPMENT <<<")
    
    # 1. Build and compile the ResNet50 model
    model = build_resnet50_model(num_classes=16)

    print("[SUCCESS] Module 3 executed. Model architecture verified.")
    print("\n" + "=" * 60)
    print("   MODULES 1, 2 & 3 COMPLETE. READY FOR MODULE 4: TRAINING  ")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
