import pandas as pd

LABELS = dict({
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
SAVE_IMG_TO = GET_IMG_FROM = 'src/train/assets/images'
MULTICHANNEL_IMGS = 'src/train/assets/images/multichannel'
SAVE_TFREC_TO = 'src/train/assets/tfrec'
SAVE_MASKS_TO = 'src/train/assets/masks'

COLORS = ['blue', 'green', 'red', 'yellow']
CELLLINES = [
    'A-431', 'A549', 'EFO-21','HAP1', 'HEK 293', 
    'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30',
    'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 
    'U-2 OS', 'U-251 MG', 'hTCEpi'
    ]

# dataframe from kaggle competition
DF = pd.read_csv('src/train/assets/kaggle_2021.tsv')
DF = DF[~DF.Label_idx.isna()]
DF_CL = DF[DF.Cellline.isin(CELLLINES)]

# number of images to download from HPA website
NUM_IMGS = 10
IMG_FORMAT = '.png'