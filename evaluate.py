import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    backbone = model.get_layer("resnet50")
    grad_model = tf.keras.models.Model(
        [backbone.inputs],
        [backbone.get_layer(last_conv_layer_name).output, backbone.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, backbone_out = grad_model(img_array)
        x = backbone_out
        for layer in model.layers:
            if layer.name == "resnet50" or isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        return np.zeros((7, 7)) # Fallback if gradient tape fails
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, save_path, alpha=0.4):
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(save_path, superimposed_img)

def main():
    print(">>> 1. INITIALIZING GPU AND LOADING MODEL <<<")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU Memory growth enabled.")
        except RuntimeError as e:
            print(e)
            
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "fighter_dataset")
    results_dir = os.path.join(base_dir, "results")
    gradcam_dir = os.path.join(results_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    
    model_path = os.path.join(results_dir, "best_model.keras")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    print(f"[SUCCESS] Loaded model from {model_path}")

    class_names = ["AV8B", "EF2000", "F14", "F15", "F16", "F18", "F35", "F4", "J10", "J20", "JF17", "KAAN", "Mig29", "Mirage2000", "Rafale", "Su57"]
    
    print("\n>>> 2. LOADING TEST DATASET <<<")
    test_dir = os.path.join(dataset_dir, "Test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=8,
        image_size=(224, 224),
        shuffle=False
    )
    file_paths = test_ds.file_paths
    
    # Preprocess test images for ResNet50
    test_ds_preprocessed = test_ds.map(
        lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    print("\n>>> 3. RUNNING INFERENCE <<<")
    preds = model.predict(test_ds_preprocessed)
    y_pred_classes = np.argmax(preds, axis=1)
    
    y_true_onehot = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_true_classes = np.argmax(y_true_onehot, axis=1)
    
    confidence = np.max(preds, axis=1)
    
    eval_results = model.evaluate(test_ds_preprocessed, verbose=0)
    loss = eval_results[0]
    accuracy = eval_results[1]
    
    top1 = tf.keras.metrics.top_k_categorical_accuracy(y_true_onehot, preds, k=1).numpy().mean()
    top5 = tf.keras.metrics.top_k_categorical_accuracy(y_true_onehot, preds, k=5).numpy().mean()
    
    macro_p = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    macro_r = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    
    weighted_p = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    weighted_r = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)

    print(f"\n[METRICS]")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"Macro Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_p:.4f} | Recall: {weighted_r:.4f} | F1: {weighted_f1:.4f}")
    
    print("\n>>> 4. GENERATING CLASSIFICATION REPORT <<<")
    report_text = classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)
    
    print("\n>>> 5. GENERATING CONFUSION MATRIX <<<")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    print("\n>>> 6. SAVING PREDICTIONS & MISCLASSIFICATION CSV <<<")
    results_df = pd.DataFrame({
        'Image Name': [os.path.basename(p) for p in file_paths],
        'True Class': [class_names[i] for i in y_true_classes],
        'Predicted Class': [class_names[i] for i in y_pred_classes],
        'Prediction Confidence': confidence,
        'Correct / Incorrect': ['Correct' if t == p else 'Incorrect' for t, p in zip(y_true_classes, y_pred_classes)],
        'File Path': file_paths
    })
    results_df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
    
    incorrect_df = results_df[results_df['Correct / Incorrect'] == 'Incorrect']
    top10_incorrect = incorrect_df.sort_values(by='Prediction Confidence', ascending=False).head(10)
    top10_incorrect.drop(columns=['File Path']).to_csv(os.path.join(results_dir, "misclassified_images.csv"), index=False)
    
    print("\n[Top 10 Misclassified]")
    print(top10_incorrect.drop(columns=['File Path']).to_string(index=False))
    
    print("\n>>> 7. GENERATING PREDICTION VISUALIZATION <<<")
    indices = random.sample(range(len(file_paths)), min(25, len(file_paths)))
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, idx in enumerate(indices):
        img = plt.imread(file_paths[idx])
        ax = axes[i // 5, i % 5]
        ax.imshow(img)
        ax.axis('off')
        true_label = class_names[y_true_classes[idx]]
        pred_label = class_names[y_pred_classes[idx]]
        conf = confidence[idx]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"T: {true_label}\nP: {pred_label} ({conf:.2f})", color=color, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_predictions.png"))
    plt.close()
    
    print("\n>>> 8. GENERATING ROC & PR CURVES <<<")
    y_true_bin = label_binarize(y_true_classes, classes=range(len(class_names)))
    
    plt.figure(figsize=(12, 10))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multiclass ROC Curve (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], preds[:, i])
        avg_pr = average_precision_score(y_true_bin[:, i], preds[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP = {avg_pr:.2f})")
    plt.title('Multiclass Precision-Recall Curve (One-vs-Rest)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pr_curve.png"))
    plt.close()
    
    print("\n>>> 9. GENERATING GRAD-CAM VISUALIZATIONS <<<")
    correct_idx = np.where(y_true_classes == y_pred_classes)[0]
    incorrect_idx = np.where(y_true_classes != y_pred_classes)[0]
    
    sel_correct = np.random.choice(correct_idx, min(10, len(correct_idx)), replace=False)
    sel_incorrect = np.random.choice(incorrect_idx, min(10, len(incorrect_idx)), replace=False)
    
    for i, idx in enumerate(sel_correct):
        img_path = file_paths[idx]
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img, axis=0))
        heatmap = make_gradcam_heatmap(img_array, model)
        save_path = os.path.join(gradcam_dir, f"correct_{i+1}_{os.path.basename(img_path)}")
        save_gradcam(img_path, heatmap, save_path)
        
    for i, idx in enumerate(sel_incorrect):
        img_path = file_paths[idx]
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img, axis=0))
        heatmap = make_gradcam_heatmap(img_array, model)
        save_path = os.path.join(gradcam_dir, f"incorrect_{i+1}_{os.path.basename(img_path)}")
        save_gradcam(img_path, heatmap, save_path)
    print(f"[SUCCESS] Grad-CAM heatmaps saved to {gradcam_dir}")

    print("\n>>> 10. MODEL ANALYSIS <<<")
    total_test = len(y_true_classes)
    correct = sum(y_true_classes == y_pred_classes)
    incorrect = total_test - correct
    avg_conf = np.mean(confidence)
    avg_conf_inc = np.mean(confidence[incorrect_idx]) if len(incorrect_idx) > 0 else 0.0

    print(f"Total Test Images: {total_test}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {incorrect}")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"Average Confidence: {avg_conf:.4f}")
    print(f"Average Incorrect Confidence: {avg_conf_inc:.4f}")

    acc_per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    hardest_idx = np.argmin(acc_per_class)
    most_accurate_idx = np.argmax(acc_per_class)
    
    print(f"Hardest Aircraft Class: {class_names[hardest_idx]} (Accuracy: {acc_per_class[hardest_idx]:.4f})")
    print(f"Most Accurate Aircraft Class: {class_names[most_accurate_idx]} (Accuracy: {acc_per_class[most_accurate_idx]:.4f})")
    
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_confused = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
    if cm_no_diag[max_confused] > 0:
        print(f"Most Confused Class Pair: {class_names[max_confused[0]]} (True) predicted as {class_names[max_confused[1]]} (Predicted)")
    else:
        print("Most Confused Class Pair: N/A")

    print("\n>>> 11. SAVING SUMMARY REPORT <<<")
    summary = f"""MODEL EVALUATION SUMMARY
========================
Total Test Images: {total_test}
Test Accuracy: {accuracy:.4f}
Test Loss: {loss:.4f}

Metrics:
- Top-1 Accuracy: {top1:.4f}
- Top-5 Accuracy: {top5:.4f}
- Precision (Weighted): {weighted_p:.4f}
- Recall (Weighted): {weighted_r:.4f}
- F1 Score (Weighted): {weighted_f1:.4f}

Analysis:
- Correct Predictions: {correct}
- Incorrect Predictions: {incorrect}
- Average Confidence: {avg_conf:.4f}
- Hardest Class: {class_names[hardest_idx]}
- Most Accurate Class: {class_names[most_accurate_idx]}
- Most Confused Pair: {class_names[max_confused[0]]} -> {class_names[max_confused[1]]}
"""
    with open(os.path.join(results_dir, "model_summary.txt"), "w") as f:
        f.write(summary)
    print(f"[SUCCESS] Saved model_summary.txt")

if __name__ == "__main__":
    main()
