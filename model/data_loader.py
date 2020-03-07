import os
from pathlib import Path
import numpy as np
import tensorflow as tf


class Dataset:
    
    def __init__(self, filepath:str, batch_size:int=32, train_test_split:float=0.95, augment:bool=False):
        self.height, self.width, self.channel = 224, 224, 3
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.augment = augment
        self.dataset_size = len(list(Path(filepath).glob('*/*')))
        self.class_names = np.array([item.name for item in Path(filepath).glob('*')])
        self.train, self.val = self.__load(filepath)
        self.wights = np.array([item.name for item in Path(filepath).glob('*')])
    
    def augment_data(self):
        """
        Applying augmentation to dataset
        """
        augmentations = [self.__flip, self.__rotate, self.__color, self.__zoom]
        
        for aug in augmentations:
            self.dataset = self.dataset.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: aug(x), lambda: x), num_parallel_calls=4)

        self.dataset = self.dataset.map(lambda x: tf.clip_by_value(x, 0, 1))
        pass
    
    def __load(self, filepath:str):        
        img_files = tf.data.Dataset.list_files(str(filepath + '*/*')).shuffle(self.dataset_size)
        print(self.dataset_size)
        print(self.class_names, self.class_names.shape)
        # print(next(iter(img_files)))
        # print 5 files
        # for f in img_files.take(5):
        #     print(f.numpy())

        train_size, val_size = int(self.train_test_split * self.dataset_size), int((1-self.train_test_split) * self.dataset_size)
        process_files_fn = lambda x: self.__process_path(x)
        dataset = img_files.map(process_files_fn, num_parallel_calls=24)

        train = dataset.take(train_size).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).prefetch(2)
        val = dataset.skip(train_size).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).prefetch(2)
        
        print(f'all: {self.dataset_size}, train: {train_size}, val: {val_size}')
        # return dataset
        return train, val

    def __process_path(self, filename:str) -> [tf.Tensor, str]:
        """Obtain the image from the filename (for both training and validation).
        The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
        """

        img_loader = tf.io.read_file(filename)
        img_decoder = tf.image.decode_jpeg(img_loader, channels=self.channel)
        img = tf.image.convert_image_dtype(img_decoder, tf.float32)
        img = tf.image.resize(img, [self.width, self.height])

        parts = tf.strings.split(filename, os.path.sep)
        print(parts[-2])
        label = tf.cast(parts[-2] == self.class_names, tf.int16)

        return img, label
    
    @staticmethod
    def __rotate(x:tf.Tensor) -> tf.Tensor:
        """
        Rotation augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        # rotate 0, 90, 180, 270 degrees
        return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, max_val=4, dtype=tf.int32))
    
    @staticmethod
    def __flip(x:tf.Tensor) -> tf.Tensor:
        """
        Flip augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x

    @staticmethod
    def __color(x:tf.Tensor) -> tf.Tensor:
        """
        Color augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
    
        # Make sure the image is still in [0, 1]
        x = tf.clip_by_value(x, 0.0, 1.0)

        return x

    @staticmethod
    def __zoom(x:tf.Tensor) -> tf.Tensor:
        """
        Zoom augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        # generate 20 crop settings, ranging from 1% to 20% crop
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
            return crops[tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.int32)]

        choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.
    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".
    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs