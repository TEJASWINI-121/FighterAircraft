import os
import tensorflow as tf

def build_resnet50_model(num_classes=16):
    """
    Constructs and compiles the ResNet50 Transfer Learning model.
    Backbone: ResNet50 (ImageNet pretrained, frozen)
    Head: BN -> Dropout(0.5) -> Dense(512) -> Dropout(0.3) -> Dense(16, softmax)
    """
    print(f"\n[INFO] Initializing ResNet50 Transfer Learning Model...")
    
    # 1. Load pretrained ResNet50 backbone
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    print(f"[SUCCESS] ResNet50 backbone loaded with ImageNet weights.")
    
    # 2. Fine Tune ResNet50 (Unfreeze last 30 layers)
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
        
    trainable_count = sum(1 for layer in backbone.layers if layer.trainable)
    frozen_count = sum(1 for layer in backbone.layers if not layer.trainable)
    print(f"[SUCCESS] Backbone fine-tuning enabled.")
    print(f"[INFO] Total trainable layers in backbone: {trainable_count}")
    print(f"[INFO] Frozen layers in backbone: {frozen_count}")

    # 3. Build classification head
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = backbone(inputs, training=False)
    
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.5, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="head_dense_512")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="head_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FighterAircraftResNet50")

    # 4. Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    
    print("[SUCCESS] Model compiled successfully.")
    return model

if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Build model
    model = build_resnet50_model(num_classes=16)
    
    # Print parameter counts
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nParameter Counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    
    print("\nModel Summary:")
    model.summary()
    
    # Save architecture image
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    arch_path = os.path.join(results_dir, "model_architecture.png")
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=arch_path,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=96
        )
        print(f"[SUCCESS] Model architecture diagram saved to: {arch_path}")
    except Exception as e:
        print(f"[WARNING] Could not save architecture diagram (Graphviz/pydot issue): {e}")

