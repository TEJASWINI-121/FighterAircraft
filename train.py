import os
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. GPU Detection and Memory Growth Configuration
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU detected.")
        print("[INFO] Memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("[INFO] GPU not detected. Using CPU.")

print('TensorFlow version:', tf.__version__)
print('Available GPU devices:', tf.config.list_physical_devices('GPU'))

from dataset.preprocessing import FighterAircraftPreprocessor
from model.resnet50_model import build_resnet50_model

def plot_training_curves(history, results_dir):
    """Plots training and validation accuracy/loss curves and saves to results directory."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "training_history.png")
    plt.savefig(plot_path)
    print(f"[INFO] Training curves saved to: {plot_path}")
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "fighter_dataset")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n>>> 2. LOADING DATASETS <<<")
    preprocessor = FighterAircraftPreprocessor(
        dataset_dir=dataset_dir, 
        batch_size=32, 
        image_size=(224, 224)
    )
    datasets = preprocessor.load_and_preprocess()

    train_ds = datasets["Train"]
    val_ds = datasets["Validation"]

    print("\n>>> 3. LOADING COMPILED RESNET50 MODEL <<<")
    # Our architecture outputs 16 classes. The loss function is already configured inside this model.
    model = build_resnet50_model(num_classes=16)

    print("\n>>> 4. CONFIGURING TRAINING CALLBACKS <<<")
    # Checkpoints
    checkpoint_path = os.path.join(results_dir, "best_model.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_best_only=True, 
        monitor="val_loss",
        verbose=1
    )
    
    # TensorBoard logging
    tensorboard_path = os.path.join(results_dir, "logs")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_path, 
        histogram_freq=1
    )

    # Early Stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    callbacks = [checkpoint_cb, tensorboard_cb, early_stopping_cb]

    print("\n>>> 5. STARTING TRAINING <<<")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=15, 
        callbacks=callbacks
    )
    print("[SUCCESS] Training completed.")

    print("\n>>> 6. PLOTTING TRAINING CURVES <<<")
    plot_training_curves(history, results_dir)

if __name__ == "__main__":
    main()
