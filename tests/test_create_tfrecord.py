from utils.create_tfrecord import CreateTFRecord

label_dict = {'a1':1, 'a2':2, 'b1':3, 'c1':4, 'c2':5, 'c3':6, 'c4':7}
CreateTFRecord(img_dir='/tests/images/', label_dict=label_dict, save_to='/tests/tfrecords/')