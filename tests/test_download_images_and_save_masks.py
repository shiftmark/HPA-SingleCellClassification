import pandas as pd
import os
from utils.download_files import DownloadFile
#from utils.get_masks import save_masks

labels = dict({
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
})

df = pd.read_csv('/content/HPA-SingleCellClassification/tests/assets/kaggle_2021.tsv')
df = df[~df.Label_idx.isna()]

colors = ['blue', 'red', 'green', 'yellow']
celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30', 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
df_17 = df[df.Cellline.isin(celllines)]
print(f'\nThere are {len(df_17)} images in the dataset.')
print(f'\nThe dataset head is: \n{df_17.head()}.')

def download(num_img=5, to_dir='/content/sample_data/hpa'):
    to_download = df_17[0:num_img]
    print(f'Attmpting to download 4 x {len(to_download)} images. One for each RGBY channel.')
    for idx, row in to_download.iterrows():
        try:
            img = row.Image       
            for c in colors:
                img_url = f'{img}_{c}.tif.gz'
                save_path = to_dir
                file_name = f'{os.path.basename(img)}_{c}'
                DownloadFile(img_url, save_path, file_name).as_image('png')
                print(f'Done: {img} - {c}')
        except:
            print(f'Failed to download {img}')

def test(from_dir='/content/sample_data/hpa', to_dir='/content/sample_data/hpa/r'):
    save_masks(from_dir, to_dir)