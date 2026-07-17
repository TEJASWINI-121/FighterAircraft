import tensorflow as tf
import os

print('--- pip show tensorflow info ---')
print('Version:', tf.__version__)
print('Build Info:', tf.sysconfig.get_build_info())
print('GPUs:', tf.config.list_physical_devices('GPU'))
