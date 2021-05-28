import sys
import os
import random
import cv2
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, '/home/adrian/Documents/HPA-SingleCellClassification/src')

from utils.download_files import DownloadFile
from utils.create_tfrecord import CreateTFRecord
from utils.parse_example import ParseExample
from train.helpers import plot_images, plot_history
from train.augment import Augment
from train.get_model import GetModel

import train.constants as c
import utils.get_masks as gm


def download(num_img=c.NUM_IMGS, to_dir=c.SAVE_IMG_TO):
    to_download = c.DF_CL[:num_img]
    print(f'Attempting to download {len(to_download)} x 4 images. One for each RGBY channel.')

    for idx, row in to_download.iterrows():
        # try:
            img = row.Image
            save_path = f'{to_dir}'#/{os.path.basename(img)}'
            # color_channels = []
            for color in c.COLORS:
                img_url = f'{img}_{color}.tif.gz'
                file_name = f'{os.path.basename(img)}_{color}'

                try:
                    DownloadFile(img_url, save_path, file_name).as_image(c.IMG_FORMAT)
                except:
                    print(f'Failed to download {img_url}.')
                
                # color_channels.append(cv2.imread(os.path.join(save_path, file_name+c.IMG_FORMAT),
                #                       cv2.IMREAD_GRAYSCALE))
            
            # if not os.path.exists(c.MULTICHANNEL_IMGS):
            #     os.makedirs(c.MULTICHANNEL_IMGS)

            # cv2.imwrite(os.path.join(c.MULTICHANNEL_IMGS,
            #                          os.path.basename(img)+c.IMG_FORMAT),
            #             cv2.merge(color_channels))
            # print(f'Multichannel image {os.path.basename(img)+c.IMG_FORMAT} saved.')
        # except:
        #     print(f'Failed to save multichannel image {os.path.basename(img)+c.IMG_FORMAT}')

def labels_df(df):
    one_hot = pd.Series(df.Label_idx).str.get_dummies()
    df = df.merge(one_hot, right_index=True, left_on=df.index)
    df.drop([
        'key_0',
        'Label',
        'Cellline',
        'in_trainset',
        'Label_idx'
        ], inplace=True, axis=1) # Keep just 'Image', '0', '1', ..., '18' columns
    
    df['Name'] = df.Image.map(lambda x: x.split('/')[-1])
    df['Labels'] = df[[str(s) for s in range(19)]].apply(lambda x: list(x), axis=1)
    
    df.drop([str(s) for s in range(19)], axis=1, inplace=True) # Drop '0', '1', ..., '18'
    df.drop(['Image'], axis=1, inplace=True) # Drop 'Image'
    df.set_index('Name', inplace=True)

    return df

# download and create tfrecord
#download(8)
#CreateTFRecord(c.MULTICHANNEL_IMGS, labels_df(c.DF)).write_to(c.SAVE_TFREC_TO, num_items_in_record=13)


#segment
gm.save_masks(c.SAVE_IMG_TO, c.SAVE_IMG_TO)

#train


#save model