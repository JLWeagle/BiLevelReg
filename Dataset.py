import random
import numpy as np
import tensorflow as tf
import glob
import os
import nibabel as nib
from pathlib import Path
from abc import ABCMeta

class BaseDataset_3D(metaclass=ABCMeta):
    """ Abstract class to flexibly utilize tf.data pipeline """

    def __init__(self, dataset_dir, train_or_val, batch_size=1, num_parallel_calls=1):
        """
        Args:
        - dataset_dir str: target dataset directory
        - train_or_val str: flag indicates train or validation
        - batch_size int: batch size
        - num_parallel_calls int: # of parallel calls
        """
        self.dataset_dir = dataset_dir
        if not train_or_val in ['train', 'val']:
            raise ValueError('train_or_val is either train or val')
        self.train_or_val = train_or_val
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls

        self._build()

    def get_samples(self):
        train_vol_names = glob.glob(os.path.join(self.dataset_dir, '*.nii.gz'))
        random.shuffle(train_vol_names)  # shuffle volume list
        self.samples = train_vol_names

    def get_atlas(self):
        atlas_vol = nib.load('data/atlas_norm_np_64.nii.gz').get_data()
        atlas_vol = atlas_vol.astype(np.float64)
        self.image_size = atlas_vol.shape
        self.atlas_vol = atlas_vol[..., np.newaxis]

    def split(self, samples):
        p = Path(self.dataset_dir)
        val_ratio = 0.05
        random.shuffle(samples)
        idx = int(len(samples) * (1 - val_ratio))
        train_samples = samples[:idx]
        val_samples = samples[idx:]

        with open(p / 'train.txt', 'w') as f: f.writelines((''.join(i) + '\n' for i in train_samples))
        with open(p / 'val.txt', 'w') as f: f.writelines((''.join(i) + '\n' for i in val_samples))

        self.samples = train_samples if self.train_or_val == 'train' else val_samples

    def parse(self, filenames):
        """
        Tensorflow file parser using native python function
        Args: tf.Tensor<tf.string> filenames: indicates images
        Returns:
        - tf.Tensor<tf.float64> image_0 : target image
        - tf.Tensor<tf.float64> image_1 : source image
        """
        return tf.py_func(self._read_py, [filenames], [tf.float64, tf.float64])

    def _read_py(self, filenames):
        """ Read image data """
        image_0 = self.atlas_vol
        image_1 = nib.load(filenames.decode()).get_data()
        image_1 = image_1.astype(np.float64)
        image_1 = image_1[..., np.newaxis]
        return image_0, image_1

    def preprocess(self, image_0, image_1):
        """ Preprocess raw images  """
        images = tf.stack([image_0, image_1], axis=0)
        images = tf.cast(images, tf.float32)
        return images

    def _build(self):
        self.get_atlas()
        self.get_samples()
        self.split(self.samples)
        self._dataset = tf.data.Dataset.from_tensor_slices(self.samples)

        if self.train_or_val == 'train':
            self._dataset = self._dataset.shuffle(len(self.samples)).repeat()

        self._dataset = (self._dataset.map(self.parse, self.num_parallel_calls)
                         .map(self.preprocess, self.num_parallel_calls)
                         .batch(self.batch_size)
                         .prefetch(1))

        return

    def get_element(self):
        """ Get data samples
        Returns:
        - images: tf.Tensor: minibatch of pairwise images

        if self.train_or_val == 'val' (i.e., validation mode), additionally returns
        - initializer: tf.data.Iterator.initializer: iterator initializer
        """
        if self.train_or_val == 'train':
            iterator = self._dataset.make_one_shot_iterator()
            images = iterator.get_next()
            images.set_shape((self.batch_size, 2, *self.image_size, 1))
            return images
        else:
            iterator = self._dataset.make_initializable_iterator()
            images = iterator.get_next()
            images.set_shape((self.batch_size, 2, *self.image_size, 1))
            return images, iterator.initializer
