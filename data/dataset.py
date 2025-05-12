import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from tifffile import imread
from skimage.transform import resize

from torchvision import transforms
import torch.nn as nn

import open_clip

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        # antibodies = pickle.load(open("/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/top_1000.pkl", "rb"))
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if fname.endswith(".tiff") or fname.endswith(".tif"):
                    path = os.path.join(fname)
                    images.append(path)

    return images

class RandomHorizontalFlip(nn.Module):
    """
    Randomly flip 3D images horizontally (along the width dimension)
    with a given probability.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (Tensor): 3D image tensor of shape (C, D, H, W) or (N, C, D, H, W)
                       where C is channels, D is depth, H is height, W is width
        Returns:
            Tensor: Randomly flipped image
        """
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[0])  # Flip along the last dimension (width)
        return x

class RandomVerticalFlip(nn.Module):
    """
    Randomly flip 3D images vertically (along the height dimension)
    with a given probability.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (Tensor): 3D image tensor of shape (C, D, H, W) or (N, C, D, H, W)
                       where C is channels, D is depth, H is height, W is width
        Returns:
            Tensor: Randomly flipped image
        """
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[1])  # Flip along the second-to-last dimension (height)
        return x

class FullFieldDataset(Dataset):
    def __init__(self, data_root = "/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/", data_len=-1, image_size=[256, 256],
                 label_dict = "/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/antibody_map.pkl",
                 annotation_dict = "/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/annotation_map.pkl",is_train=True):
        flist = make_dataset(data_root)
        if data_len > 0:
            idx = np.random.choice(len(flist), data_len, replace=False)
            self.flist = [flist[i] for i in idx]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
            torch.from_numpy,
            RandomHorizontalFlip(p=0.25),
            RandomVerticalFlip(p=0.25),
        ])
        self.tfs_no_flip = transforms.Compose([
            torch.from_numpy,
        ])
        # self.test_data = pd.read_csv("/scratch/groups/emmalu/Beacons/test_set.csv", header=None)
        self.is_train = is_train
        self.img_shape = image_size
        self.data_root = data_root
        #_, preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.clip_image_preprocess = transform = transforms.Compose([
                    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]
                    )
                    ])
        


        self.cell_line_dict = {'A-431': 0, 'SiHa': 1, 'U2OS': 2, 'U-251MG': 3, 'RT-4': 4, 'BJ [Human fibroblast]': 5, 'MCF-7': 6, 
                               'HeLa': 7, 'SH-SY5Y': 8, 'GAMG': 9, 'A-549': 10, 'Rh30': 11, 'SK-MEL-30': 12, 'PC-3': 13, 'hTCEpi': 14, 
                               'HEK293': 15, 'HEL': 16, 'Hep-G2': 17, 'CACO-2': 18, 'OE19': 19, 'NIH 3T3': 20, 'JURKAT': 21, 'HaCaT': 22, 
                               'AF22': 23, 'SuSa': 24, 'THP-1': 25, 'REH': 26, 'EFO-21': 27, 'hTERT-RPE1': 28, 'ASC52telo': 29, 
                               'K-562': 30, 'HDLM-2': 31, 'HAP1': 32, 'HBEC3-KT': 33, 'NB4': 34, 'LHCN-M2': 35, 'PODO SVTERT152': 36, 
                               'PODO TERT256': 37, 'HUVEC TERT2': 38, 'RPTEC TERT1': 39}

        self.label_dict = pickle.load(open(label_dict, "rb"))
        self.annotation_dict = pickle.load(open(annotation_dict, "rb"))
    
    def __len__(self):
        
        return len(self.flist)
    
    def __getitem__(self, idx):
        ret = {}
        file_name = str(self.flist[idx]).zfill(5)
        img = imread('{}/{}'.format(self.data_root, file_name))

        img *= 2
        img -= 1

        if self.is_train:
            img = self.tfs(img)
        else:
            img = self.tfs_no_flip(img)
        img_clip = (img[:, :, [0, 2, 3]]+1)/2
        img_clip = self.clip_image_preprocess(img_clip.permute(2, 0, 1))
        #crop to 224*224
        #img = img[16:-16,16:-16,:]

        gt_img = img[:, :, [1]]
        cond_img = img[:, :, [0, 2, 3]]

        # move to channel first
        gt_img = torch.permute(gt_img, (2, 0, 1))
        cond_img = torch.permute(cond_img, (2, 0, 1))
        img = torch.permute(img, (2, 0, 1))
        cond_img_size = cond_img.size()
        assert cond_img_size[0] == 3, "Condition image should have 3 channels"
        assert cond_img_size[1] == 256, "Condition image should have 256 height"
        assert cond_img_size[2] == 256, "Condition image should have 256 width"

        cell_line = file_name.split('_')[0]
        ab = file_name.split('_')[1]

        ret = {}
        ret['gt_image'] = gt_img
        ret['cond_image'] = cond_img
        
        ret['image'] = img
        
        ret['label'] = self.label_dict[ab]
        ret['protein_name'] = ab
        ret['cell_line'] = self.cell_line_dict[cell_line]
        ret['cell_line_name'] = cell_line
        ret['annotation'] = torch.from_numpy(np.array(self.annotation_dict[ab][cell_line]))
        ret['path'] = file_name

        ret["clip_image"] = img_clip
        return ret