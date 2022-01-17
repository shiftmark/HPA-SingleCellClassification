"""Save masks using CellSegmentation"""
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell#, label_nuclei
from typing import List
import glob
import os
import imageio
 
NUC_MODEL = 'nucleai-model.pth'  # will be downloaded if it doesn't exist
CELL_MODEL = 'cell-model.pth'  # will be downloaded if it doesn't exist
SEGMENTATOR = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor = .25,
    device = 'cuda',
    padding = False,
    multi_channel_model=True
)

def save_masks(from_dir, to_dir, save_cell_mask=True, save_nuc_mask=True):
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    microtubule: List[str] = glob.glob(from_dir + '/' + '*_red.png')
    endo_ret: List[str] = [e.replace('red', 'yellow') for e in microtubule]
    nuclei: List[str] = [n.replace('red', 'blue') for n in microtubule]
    images: List[List[str]] = [microtubule, endo_ret, nuclei]

    nuc_segmentations = SEGMENTATOR.pred_nuclei(images[2])
    cell_segmetnations = SEGMENTATOR.pred_cells(images)

    for idx, predictions in enumerate(cell_segmetnations):
        nuc_mask, cell_mask = label_cell(nuc_segmentations[idx], cell_segmetnations[idx])
        if save_cell_mask:
            cell_mask_name = os.path.basename(microtubule[idx]).replace('red', 'cell_mask')
            imageio.imwrite(os.path.join(to_dir, cell_mask_name), cell_mask)
        if save_nuc_mask:
            nuc_mask_name = os.path.basename(microtubule[idx]).replace('red', 'nuc_mask')
            imageio.imwrite(os.path.join(to_dir, nuc_mask_name), nuc_mask)
