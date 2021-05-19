import os
import tensorflow as tf

class CreateTFRecord:
    """
    Creates a record.tfrec file from images in a directory
    features:
        img_name - String from filename.split('.')[0];
        img - Array from png encoded image data;
        label - String from dictionary {img_name: label}.
    Args:
        img_dir (str) - path to images folder.
        label_dict (dict) - dictionary containig {img_name: label} value pairs.

    """

    def __init__(self, img_dir, label_dict):
        self.img_dir = img_dir
        self.label_dict = label_dict 
        self.files = [os.listdir(self.img_dir)]
        
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64=tf.train.Int64List(value=[value]))

    @staticmethod
    def _image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()]))

    @classmethod
    def serialize(cls, img_name, img, label):
        """
        Serialize img_name, img, label to byte string.
        Args:
            img_name (bytes list) - Image name.
            img (bytes list) - Image data.
            label (int list) - Label
        Returns SerializeToString object.
        """
        features = {
            'img_name': cls._bytes_feature(img_name),
            'img': cls._image_feature(img),
            'label': cls._int64_feature(label)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def write_to(self, save_to='./'):
        """
        Writes serialized data to TFRecord named recprd.tfrec.
        Args: 
            save_to (str) - Path to destination folder. Default: './'
        Returns: None - saves the file to location.
        """
        writer = tf.io.TFRecordWriter(save_to + 'record.tfrec')
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        with writer as w:
            for file in self.files:
                img_name = file.split('.')[0]
                img = tf.io.decode_png(tf.io.read_file(f'{self.img_dir}{file}'))
                label = self.label_dict[img_name]
                serialized = self.serialize(img_name, img, label)
                w.write(serialized)