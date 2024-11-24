import tensorflow as tf

# This checks if a GPU is available
if tf.test.is_gpu_available():
    print("A GPU is available.")
else:
    print("No GPU available.")
