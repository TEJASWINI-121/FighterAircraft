import numpy as np

def preprocess_input(x, data_format=None):
    """
    Simulates tf.keras.applications.resnet50.preprocess_input.
    Preprocesses a batch of double/float image tensors for ResNet50:
    - Converts RGB to BGR.
    - Subtracts ImageNet channel means: [103.939, 116.779, 123.68].
    
    Args:
        x (numpy.ndarray): Float32 images of shape (batch, 224, 224, 3) in [0, 255].
        
    Returns:
        numpy.ndarray: Preprocessed images.
    """
    x_out = np.array(x, dtype=np.float32)
    # Swap Channel 0 (Red) and Channel 2 (Blue) to convert RGB -> BGR
    r = x_out[..., 0].copy()
    b = x_out[..., 2].copy()
    x_out[..., 0] = b
    x_out[..., 2] = r
    
    # Subtract ImageNet BGR mean
    x_out[..., 0] -= 103.939  # Blue mean
    x_out[..., 1] -= 116.779  # Green mean
    x_out[..., 2] -= 123.68   # Red mean
    return x_out
