import tensorflow as tf
from typing import Tuple, List, Union, Any

class ParseExample:
    """Parse an item serialized dataset.
    features: img_name - tf.io.FixedLenFeature (string)
              img - tf.io.FixedLenFeature (string) - png encoded
              label - tf.io.FixedLenFeature (int64)
    Returns img_name if return_img_name is True
    """
    def __init__(self, example, return_img_name=False):
        self.example = example
        self.return_img_name = return_img_name

    def parse_eg(self) -> Union[Tuple[Any, Any], Tuple[Any, Any, Any]]:
        features = {
            'img_name': tf.io.FixedLenFeature([], tf.string),
            'img': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        eg = tf.io.parse_single_example(self.example, features)
        img_name = eg['img_name']
        img = tf.image.decode_png(eg['img'])
        label = tf.io.decode_raw(eg['label'], out_type=tf.uint8)

        if self.return_img_name:
            return img_name, img, label
        return img, label
