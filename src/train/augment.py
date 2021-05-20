import tensorflow as tf

class Augment:
    """
    Apply augmentation on the a dataset via .map method from tf.data.Dataset API.
    Args:
        brightness (list or None) - Default None. Keyward arguments for tf.image.stateless_random_brightness;
        contrast (list or None) - Default None. Keyward arguments for tf.image.stateless_random_contrast;
        flip_horizontal (bool) - Default False. If True, randomly flip an image horizontally (left to right) deterministically;
        flip_vertical (bool) - Default False. If True, randomly flip an image vertically (up to down) deterministically;
        hue (dict or None) - Default None. Keyward arguments for tf.image.stateless_random_hue; 
        rotate (list or None) - Default None. Keyward arguments for tf.image.rot90 (rotate image counter-clockwise by 90 degrees);
        saturation (list or None) - Default None. Keyward arguments for tf.image.stateless_random_saturation;
        seed (shape 2 Tensor or tuple of 2 ints) - Guarantees the same results given the same seed independent of how many times the function is called. 
    """

    def __init__(self, image, label=None, seed=(0,0)):
        self.seed = seed
        self.image = image
        self.label = label

    def brightness(self, *args, **kwargs):
        self.image = tf.image.stateless_random_brightness(*args, **kwargs, seed=self.seed)
        return self

    def contrast(self, *args, **kwargs):
        self.image = tf.image.stateless_random_contrast(*args, **kwargs, seed=self.seed)
        return self

    def flip_horizontal(self, *args, **kwargs):
        self.image = tf.image.stateless_random_flip_left_right(*args, **kwargs, seed=self.seed)
        return self

    def flip_vertical(self, *args, **kwargs):
        self.image =  tf.image.stateless_random_flip_up_down(*args, **kwargs, seed=self.seed)
        return self

    def hue(self, *args, **kwargs):
        self.image = tf.image.stateless_random_hue(*args, **kwargs, seed=self.seed)
        return self

    def rotate(self, *args, **kwargs):
        self.image = tf.image.rot90(*args, **kwargs)
        return self

    def saturation(self, *args, **kwargs):
        self.image = tf.image.stateless_random_saturation(*args, **kwargs, seed=self.seed)
        return self

    def __call__(self):
        if self.label is not None:
            return self.image, self.label
        return self.image





    



    
