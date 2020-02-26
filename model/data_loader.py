import numpy as np
import tensorflow as tf


class DataLoader:
    
    def __init__(self, filepath:str, augment:bool=False):
        self.filepath = filepath
        self.dataset = self.__load(filepath)
    
    def __load(self, fliepath:str) -> tf.data.Dataset:

        pass
    
    def augment(self):
        """
        Applying augmentation to dataset
        """
        augmentations = [self.__flip, self.__rotate, self.__color, self.__zoom]
        
        for aug in augmentations:
            self.dataset = self.dataset.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: aug(x), lambda: x), num_parallel_calls=4)

        self.dataset = self.dataset.map(lambda x: tf.clip_by_value(x, 0, 1))
        pass

    def __rotate(self, x:tf.Tensor) -> tf.Tensor:
        """
        Rotation augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        # rotate 0, 90, 180, 270 degrees
        return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, max_val=4, dtype=tf.int32))

    def __flip(self, x:tf.Tensor) -> tf.Tensor:
        """
        Flip augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x

    def __color(self, x:tf.Tensor) -> tf.Tensor:
        """
        Color augmentation
        
        Inputs: x: tf.Tensor image
        Outputs: tf.Tensor image
        """
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)

        return x

    def __zoom(self, x:tf.Tensor) -> tf.Tensor:
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

        return tf.cond(choice < 0.5, lambda x, lambda: random_crop(x))

    


def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label


def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.
    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


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