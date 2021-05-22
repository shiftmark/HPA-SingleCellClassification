import tensorflow as tf

class Augment():
    """
    Apply augmentation on the a dataset via .map method from tf.data.Dataset API.
    Args:
        brightness (dict or None) - Default None. Keyward arguments for tf.image.stateless_random_brightness;
        contrast (dict or None) - Default None. Keyward arguments for tf.image.stateless_random_contrast;
        flip_horizontal (bool) - Default False. If True, randomly flip an image horizontally (left to right) deterministically;
        flip_vertical (bool) - Default False. If True, randomly flip an image vertically (up to down) deterministically;
        hue (dict or None) - Default None. Keyward arguments for tf.image.stateless_random_hue;
        rotate (dict or None) - Default None. Keyward arguments for tf.image.rot90 (rotate image counter-clockwise by 90 degrees);
        saturation (dict or None) - Default None. Keyward arguments for tf.image.stateless_random_saturation;
        seed (shape 2 Tensor or tuple of 2 ints) - Guarantees the same results given the same seed independent of how many times the function is called. 
    """

    def __init__(self, brightness=None, contrast=None, flip_horizontal=False, flip_vertical=False, hue=None, rotate=None, saturation=None, seed=(0,0)):
        self.brightness = brightness
        self.contrast = contrast
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.hue = hue
        self.rotate = rotate
        self.saturation = saturation
        self.seed = seed

    def _brightness(self, image, **kwargs):
        return tf.image.stateless_random_brightness(image, **kwargs, seed=self.seed)

    def _contrast(self, image, **kwargs):
        return tf.image.stateless_random_contrast(image, **kwargs, seed=self.seed)

    def _flip_horizontal(self, image):
        return tf.image.stateless_random_flip_left_right(image, seed=self.seed)

    def _flip_vertical(self, image):
        return tf.image.stateless_random_flip_up_down(image, seed=self.seed)

    def _hue(self, image, **kwargs):
        return tf.image.stateless_random_hue(image, **kwargs, seed=self.seed)

    def _rotate(self, image, **kwargs):
        return tf.image.rot90(image, **kwargs)

    def _saturation(self, image, **kwargs):
        return tf.image.stateless_random_saturation(image, **kwargs, seed=self.seed)

    def _augment(self, image, label=None):
        image = self._brightness(image, **self.brightness) if self.brightness else image
        image = self._contrast(image, **self.contrast) if self.contrast else image
        image = self._flip_horizontal(image) if self.flip_horizontal else image
        image = self._flip_vertical(image) if self.flip_vertical else image
        image = self._hue(image, **self.hue) if self.hue else image
        image = self._rotate(image, **self.rotate) if self.rotate else image
        image = self._saturation(image, **self.saturation) if self.saturation else image

        if label is not None:
            return image, label
        return image

    def apply_on(self, dataset):
        """
        Applies the augmentation on a dataset of (image, label) or (image) objects.
        Args:
            dataset (dataset) - The tf.data.Dataset object.
        Returns:
            dataset.map object
        """
        return dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        