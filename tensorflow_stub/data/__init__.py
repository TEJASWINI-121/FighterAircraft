import random
import numpy as np

AUTOTUNE = -1

class Tensor(np.ndarray):
    """
    A subclass of numpy ndarray that mimics a TensorFlow Tensor by providing 
    a .numpy() method.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
        
    def numpy(self):
        return np.asarray(self)

class Dataset:
    """
    A class that simulates tf.data.Dataset pipeline behaviors including mapping,
    caching, batching, and prefetching.
    """
    def __init__(self, items):
        self.items = items
        self._batch_size = 1
        self._shuffle = False
        self._cache = False

    def cache(self):
        self._cache = True
        return self

    def shuffle(self, buffer_size, seed=None):
        self._shuffle = True
        if seed is not None:
            random.seed(seed)
        return self

    def batch(self, batch_size):
        self._batch_size = batch_size
        return self

    def prefetch(self, buffer_size):
        return self

    def map(self, map_func, num_parallel_calls=None):
        new_items = []
        for x, y in self.items:
            res_x, res_y = map_func(x, y)
            # Ensure return is unwrapped back to numpy array if a tensor subclass was used
            new_items.append((np.asarray(res_x), np.asarray(res_y)))
        self.items = new_items
        return self

    def take(self, count):
        batched_items = []
        n = len(self.items)
        items_to_use = list(self.items)
        if self._shuffle:
            random.shuffle(items_to_use)
            
        for i in range(0, n, self._batch_size):
            chunk = items_to_use[i:i + self._batch_size]
            if not chunk:
                break
            batch_x = np.array([x for x, y in chunk], dtype=np.float32)
            batch_y = np.array([y for x, y in chunk], dtype=np.int32)
            batched_items.append((Tensor(batch_x), Tensor(batch_y)))
            if len(batched_items) == count:
                break
        return batched_items

    def __len__(self):
        return int(np.ceil(len(self.items) / self._batch_size))

    def __iter__(self):
        items_to_use = list(self.items)
        if self._shuffle:
            random.shuffle(items_to_use)
        n = len(self.items)
        for i in range(0, n, self._batch_size):
            chunk = items_to_use[i:i + self._batch_size]
            batch_x = np.array([x for x, y in chunk], dtype=np.float32)
            batch_y = np.array([y for x, y in chunk], dtype=np.int32)
            yield (Tensor(batch_x), Tensor(batch_y))
