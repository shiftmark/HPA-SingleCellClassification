import os
import tensorflow as tf

class CreateTFRecord:
    '''
    Creates a record.tfrec file from images in a directory
    features:
        img_name - String from filename.split('.')[0]
        img - Array from png encoded image data
        label - String from dictionary {img_name: label}
    '''

    def __init__(self, img_dir, label_dict, save_to='./'):
        self.img_dir = img_dir
        self.label_dict = label_dict
        self.save_to = save_to
        self.files = [os.listdir(self.img_dir)]
        if not os.path.exists(self.save_to):
            os.makedirs(self.save_to)
        self.writer = tf.io.TFRecordWriter(self.save_to + 'record.tfrec')


    def _bytes_feature(self, value):
        '''Returns a bytes_list from a string.'''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def _float_feature(self, value):
        '''Returns a float_list from a float.'''
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        '''Returns an int64_list from a bool / enum / int / uint.'''
        return tf.train.Feature(int64=tf.train.Int64List(value=[value]))

    def _image_feature(self, value):
        '''Returns a bytes_list from a string / byte.'''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()]))

    def serialize(self, img_name, img, label):
        features = {
            'img_name': self._bytes_feature(img_name),
            'img': self._image_feature(img),
            'label': self._label_feature(label)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def writer(self):
        with self.writer as writer:
            for file in self.files:
                img_name = file.split('.')[0]
                img = tf.io.decode_png(tf.io.read_file(f'{self.img_dir}{file}'))
                label = self.label_dict[img_name]
                serialized = self.serialize(img_name, img, label)
                writer.write(serialized)