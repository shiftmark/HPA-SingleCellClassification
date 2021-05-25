import os
import tensorflow as tf
import pandas as pd

class CreateTFRecord:
    """
    Creates a record.tfrec file from images in a directory using labels from:
    1. a {name:label} dictionary OR
    2. a dataframe with image names in index column and lables in column named 'Labels'.
        features:
            img_name - String from filename.split('.')[0];
            img - Array from png encoded image data;
            label - String from dictionary {img_name: label}.
    Images should be the same size.

    Args:
        img_dir (str) - Path to images folder.
        labels (dict or pd.DataFrame) - Dictionary or dataframe containig {img_name: label} value pairs.
    """

    def __init__(self, img_dir, labels):
        self.img_dir = img_dir
        self.labels = labels 
        self.files = os.listdir(self.img_dir)

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()]))

    @staticmethod
    def _one_hot_feature(value):
        """Returns a bytes_list from a list of values."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value)]))

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
            'label': cls._one_hot_feature(label)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def write_to(self, save_to='.', num_items_in_record=10):
        """
        Writes serialized data to a specified number of tfrecords, named 'tfrecord{number}.tfrec'.
        Args: 
            save_to (str) - Path to destination folder. Default: '.'
            num_items_in_record - The number if items (img_name, img and/or label) to write in a record. Default: 10
        Returns: None - saves the file to location.
        """
        
        if not os.path.exists(save_to):
           os.makedirs(save_to)
        
        num_files = len(self.files)
        num_records = num_files//num_items_in_record + int(num_files % num_items_in_record != 0)
        
        for rec in range(num_records):
            writer = tf.io.TFRecordWriter(save_to + f'/record{rec}.tfrec')
            ct2 = min(num_items_in_record, num_files-rec*num_items_in_record)

            with writer as w:
                for i in range(ct2):
                    fl = self.files[num_items_in_record*rec + i]

                    img_name = fl.split('.')[0]
                    img = tf.io.decode_png(tf.io.read_file(f'{self.img_dir}/{fl}'))
                    if isinstance(self.labels, dict):
                        label = self.labels[img_name]
                    elif isinstance(self.labels, pd.DataFrame):
                        label = self.labels.at[img_name, 'Labels']

                    serialized = self.serialize(img_name, img, label)
                    w.write(serialized)
    def test(self, save_to='.', num_items_in_record=10):
        """
        Prints:
        1. Number of items to write;
        2. Number of records to write the files in;
        3. Number of items in each record;
        4. The item in each record;
        Args: 
            save_to (str) - Path to destination folder. Default: '.'
            num_items_in_record - The number if items (img_name, img and/or label) to write in a record. Default: 10
        Returns: None - prints information.

        """
        
        if not os.path.exists(save_to):
           os.makedirs(save_to)
        
        num_files = len(self.files)
        print(f'Number of files: {num_files}')
        num_records = num_files//num_items_in_record + int(num_files % num_items_in_record != 0)
        print(f'Number of records: {num_records}')
        for rec in range(num_records):
            
            ct2 = min(num_items_in_record, num_files-rec*num_items_in_record)
            print(f'Number of items: {ct2}, in record: {rec}')

            
            for i in range(ct2):
                fl = self.files[num_items_in_record*rec + i]
                print(f'Item: {fl}')

