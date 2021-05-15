import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell#, label_nuclei
import glob
import os
import imageio
 
NUC_MODEL = 'nucleai-model.pth' # will be downloaded if doesn't exist
CELL_MODEL = 'cell-model.pth' # will be downloaded if doesn't exist
SEGMENTATOR = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor = .25,
    device = 'cuda',
    padding = False,
    multi_channel_model=True
)

class GetMasks:
    def __init__(self, from_dir, to_dir, save_cell_mask=True, save_nuc_mask=True):
        self.from_dir = from_dir
        self.to_dir = to_dir
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        self.save_cell_mask = save_cell_mask
        self.save_nuc_mask = save_nuc_mask
        
        self.microtubule = glob.glob(self.from_dir + '/' + '*_red.png')
        self.endo_ret = [e.replace('red', 'yellow') for e in self.microtubule]
        self.nuclei = [n.replace('red', 'blue') for n in self.microtubule]
        
        self.images = [self.microtubule, self.endo_ret, self.nuclei]
        
        self.nuc_segmentations = SEGMENTATOR.pred_nuclei(self.images[2])
        self.cell_segmetnations = SEGMENTATOR.pred_cells(self.images)

    def __call__(self, *args, **kwds):
        print(len(self.cell_segmetnations), len(self.nuc_segmentations))
        for idx, prediction in enumerate(self.cell_segmetnations):
            nuc_mask, cell_mask = label_cell(self.nuc_segmentations[idx], self.cell_segmetnations[idx])
            if self.save_cell_mask:
                cell_mask_name = os.path.basename(self.microtubule[idx]).replace('red', 'cell_mask')
                imageio.imwrite(os.path.join(self.to_dir, cell_mask_name), cell_mask)
            if self.save_nuc_mask:
                nuc_mask_name = os.path.basename(self.microtubule[idx]).replace('red', 'nuc_mask')
                imageio.imwrite(os.path.join(self.to_dir, nuc_mask_name), nuc_mask)
            

GetMasks('/home/adrian/HPA-SingleCellClassification/HPA-SingleCellClassification/src/train/images', '/home/adrian/HPA-SingleCellClassification/HPA-SingleCellClassification/src/train/images')