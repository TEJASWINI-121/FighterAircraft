# Fighter Aircraft Recognition

## Project Overview
This project aims to develop a deep learning-based aircraft recognition system. The system can classify various fighter aircraft using a fine-grained dataset and a convolutional neural network.

## Objectives
* Load and preprocess a robust fighter aircraft dataset.
* Develop an image classification model to recognize different aircraft.
* Validate and evaluate the model's performance on recognizing subtle fine-grained features.
* Provide an end-to-end ML pipeline with a predictable and organized structure.

## Folder Structure
```text
dataset/            # Dataset loader and related utilities
model/              # Deep learning model definition (e.g., ResNet50 formulation)
results/            # Model evaluation results, plots, and metrics
dataset_files/      # The raw dataset and processed images (not tracked via git)
main.py             # Main entry point for the project
verify_curation.py  # Utility to verify dataset integrity and structure
curate_dataset.py   # Dataset preparation and curation script
requirements.txt    # Python dependencies
README.md           # Project documentation
```

## Technologies Used
* **Language**: Python 3
* **Framework**: TensorFlow / Keras
* **Libraries**: OpenCV, NumPy, Matplotlib, scikit-learn

## Dataset Description
The system is built to process images of various fighter jets (e.g., F-16, F-22, F-35, MiG-29, Su-30, Rafale). It parses the dataset structure (which includes standard annotations if available) and ensures class balance.

## Completed Modules
- [x] **Module 1**: Dataset Preparation & Directory Organization
- [x] **Module 2**: Dataset Curation & Loading (handling imbalanced data, verifying images)
- [x] **Module 3**: Model Development (Architecting the ResNet50 model)

## Current Project Status
**Module 3 has been successfully completed.** This entails the complete implementation of the ResNet50 model scaffolding, data curation, and dataset verification. The project repository is now structured and fully prepared for training.

## Future Modules
- [ ] **Module 4**: Training the Model
- [ ] **Module 5**: Model Evaluation and Metrics
- [ ] **Module 6**: Prediction & Inference Pipeline
- [ ] **Module 7**: Frontend / UI Development

## Setup Instructions

### How to Create a Virtual Environment
```bash
python -m venv .venv
```

### How to Install Dependencies
Activate the virtual environment:
* **Windows**: `.venv\Scripts\activate`
* **macOS/Linux**: `source .venv/bin/activate`

Then install the necessary packages:
```bash
pip install -r requirements.txt
```

### How to Run the Project
To verify the dataset format and view sample curation output:
```bash
python main.py
```
*(Additional modules will provide distinct execution targets once implemented.)*
