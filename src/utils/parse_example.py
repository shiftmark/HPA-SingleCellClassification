import tensorflow as tf

class ParseExample:
    '''
    Parse a tfrecord from path.
    features: img_name - tf.io.FixedLenFeature (string)
              img - tf.io.FixedLenFeature (string) - png encoded
              label - tf.io.FixedLenFeature (int64)
    Returns img_name if return_img_name is True
    '''
    def __init__(self, example, return_img_name=False):
        self.example = example
        self.return_img_name = return_img_name

    def parse_eg(self):
        features = {
            'img_name': tf.io.FixedLenFeature([], tf.string),
            'img': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        eg = tf.io.parse_single_example(self.example, features)
        img_name = eg['img_name']
        img = tf.image.decode_png(eg['img'])
        label = eg['label']

        if self.return_img_name:
            return img_name, img, label
        return img, label
        