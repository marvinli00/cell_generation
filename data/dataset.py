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


# class FullFieldDataset(Dataset):
#     def __init__(self, data_root="/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/", 
#                  data_len=-1, image_size=[256, 256],
#                  label_dict="/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/antibody_map.pkl",
#                  annotation_dict="/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/annotation_map.pkl",
#                  is_train=True, validate_files=True):
        
#         flist = make_dataset(data_root)
        
#         # Validate files if requested
#         if validate_files:
#             flist = self._validate_files(flist, data_root)
#             print(f"Valid files after validation: {len(flist)}")
        
#         if data_len > 0:
#             idx = np.random.choice(len(flist), min(data_len, len(flist)), replace=False)
#             self.flist = [flist[i] for i in idx]
#         else:
#             self.flist = flist
            
#         # Rest of your initialization code...
#         self.tfs = transforms.Compose([
#             torch.from_numpy,
#             RandomHorizontalFlip(p=0.25),
#             RandomVerticalFlip(p=0.25),
#         ])
#         self.tfs_no_flip = transforms.Compose([
#             torch.from_numpy,
#         ])
        
#         self.is_train = is_train
#         self.img_shape = image_size
#         self.data_root = data_root
        
#         self.clip_image_preprocess = transforms.Compose([
#             transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#             transforms.Normalize(
#                 mean=[0.48145466, 0.4578275, 0.40821073],
#                 std=[0.26862954, 0.26130258, 0.27577711]
#             )
#         ])
        
#         self.cell_line_dict = {'A-431': 0, 'SiHa': 1, 'U2OS': 2, 'U-251MG': 3, 'RT-4': 4, 'BJ [Human fibroblast]': 5, 'MCF-7': 6, 
#                                'HeLa': 7, 'SH-SY5Y': 8, 'GAMG': 9, 'A-549': 10, 'Rh30': 11, 'SK-MEL-30': 12, 'PC-3': 13, 'hTCEpi': 14, 
#                                'HEK293': 15, 'HEL': 16, 'Hep-G2': 17, 'CACO-2': 18, 'OE19': 19, 'NIH 3T3': 20, 'JURKAT': 21, 'HaCaT': 22, 
#                                'AF22': 23, 'SuSa': 24, 'THP-1': 25, 'REH': 26, 'EFO-21': 27, 'hTERT-RPE1': 28, 'ASC52telo': 29, 
#                                'K-562': 30, 'HDLM-2': 31, 'HAP1': 32, 'HBEC3-KT': 33, 'NB4': 34, 'LHCN-M2': 35, 'PODO SVTERT152': 36, 
#                                'PODO TERT256': 37, 'HUVEC TERT2': 38, 'RPTEC TERT1': 39}

#         self.label_dict = pickle.load(open(label_dict, "rb"))
#         self.annotation_dict = pickle.load(open(annotation_dict, "rb"))
        
#         # Keep track of failed files for logging
#         self.failed_files = set()

#     def _validate_files(self, flist, data_root):
#         """Pre-validate TIFF files to remove corrupted ones"""
#         valid_files = []
#         corrupted_count = 0
        
#         print("Validating TIFF files...")
#         for i, file_name in enumerate(flist):
#             if i % 1000 == 0:
#                 print(f"Validated {i}/{len(flist)} files, found {corrupted_count} corrupted")
                
#             try:
#                 file_path = f'{data_root}/{str(file_name).zfill(5)}'
#                 img = imread(file_path)
                
#                 # Check basic properties
#                 if img is None or img.size == 0:
#                     raise ValueError("Empty image")
                    
#                 # Check expected shape (adjust as needed)
#                 if len(img.shape) != 3 or img.shape[2] != 4:
#                     raise ValueError(f"Unexpected shape: {img.shape}")
                    
#                 valid_files.append(file_name)
                
#             except Exception as e:
#                 corrupted_count += 1
#                 logging.warning(f"Corrupted file {file_name}: {e}")
                
#         print(f"Validation complete: {len(valid_files)} valid, {corrupted_count} corrupted")
#         return valid_files

#     def _get_fallback_sample(self, original_idx):
#         """Get a fallback sample when the original fails"""
#         # Try a few random samples
#         for _ in range(5):
#             try:
#                 fallback_idx = np.random.randint(0, len(self.flist))
#                 if fallback_idx != original_idx:
#                     return self._load_sample(fallback_idx, is_fallback=True)
#             except:
#                 continue
        
#         # If all else fails, return a dummy sample
#         return self._get_dummy_sample()

#     def _get_dummy_sample(self):
#         """Create a dummy sample as last resort"""
#         dummy_img = torch.zeros(4, 256, 256)
#         dummy_img_clip = torch.zeros(3, 224, 224)
        
#         # Use first valid cell line and protein
#         first_cell_line = list(self.cell_line_dict.keys())[0]
#         first_ab = list(self.label_dict.keys())[0]
        
#         ret = {
#             'gt_image': dummy_img[1:2],
#             'cond_image': dummy_img[[0,2,3]],
#             'image': dummy_img,
#             'label': self.label_dict[first_ab],
#             'protein_name': first_ab,
#             'cell_line': self.cell_line_dict[first_cell_line],
#             'cell_line_name': first_cell_line,
#             'annotation': torch.zeros(10),  # Adjust size as needed
#             'path': 'dummy',
#             'clip_image': dummy_img_clip
#         }
#         return ret

#     def _load_sample(self, idx, is_fallback=False):
#         """Load a single sample with error handling"""
#         file_name = str(self.flist[idx]).zfill(5)
        
#         try:
#             img = imread(f'{self.data_root}/{file_name}')
            
#             # Check if image loaded properly
#             if img is None or img.size == 0:
#                 raise ValueError("Empty or corrupted image")
            
#             img = img.astype(np.float32)  # Ensure proper dtype
#             img *= 2
#             img -= 1

#             if self.is_train:
#                 img = self.tfs(img)
#             else:
#                 img = self.tfs_no_flip(img)
                
#             img_clip = (img[:, :, [0, 2, 3]] + 1) / 2
#             img_clip = self.clip_image_preprocess(img_clip.permute(2, 0, 1))

#             gt_img = img[:, :, [1]]
#             cond_img = img[:, :, [0, 2, 3]]

#             # Move to channel first
#             gt_img = torch.permute(gt_img, (2, 0, 1))
#             cond_img = torch.permute(cond_img, (2, 0, 1))
#             img = torch.permute(img, (2, 0, 1))
            
#             # Validate shapes
#             cond_img_size = cond_img.size()
#             assert cond_img_size[0] == 3, f"Condition image should have 3 channels, got {cond_img_size[0]}"
#             assert cond_img_size[1] == 256, f"Condition image should have 256 height, got {cond_img_size[1]}"
#             assert cond_img_size[2] == 256, f"Condition image should have 256 width, got {cond_img_size[2]}"

#             cell_line = file_name.split('_')[0]
#             ab = file_name.split('_')[1]

#             ret = {
#                 'gt_image': gt_img,
#                 'cond_image': cond_img,
#                 'image': img,
#                 'label': self.label_dict[ab],
#                 'protein_name': ab,
#                 'cell_line': self.cell_line_dict[cell_line],
#                 'cell_line_name': cell_line,
#                 'annotation': torch.from_numpy(np.array(self.annotation_dict[ab][cell_line])),
#                 'path': file_name,
#                 'clip_image': img_clip
#             }
            
#             return ret
            
#         except Exception as e:
#             if not is_fallback:
#                 if file_name not in self.failed_files:
#                     self.failed_files.add(file_name)
#                     logging.warning(f"Failed to load {file_name}: {e}")
#             raise e

#     def __getitem__(self, idx):
#         try:
#             return self._load_sample(idx)
#         except Exception:
#             # Try fallback
#             try:
#                 return self._get_fallback_sample(idx)
#             except Exception:
#                 # Last resort - return dummy
#                 logging.error(f"All fallbacks failed for index {idx}, returning dummy sample")
#                 return self._get_dummy_sample()

#     def __len__(self):
#         return len(self.flist)

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
        


        # self.cell_line_dict = {'A-431': 0, 'SiHa': 1, 'U2OS': 2, 'U-251MG': 3, 'RT-4': 4, 'BJ [Human fibroblast]': 5, 'MCF-7': 6, 
        #                        'HeLa': 7, 'SH-SY5Y': 8, 'GAMG': 9, 'A-549': 10, 'Rh30': 11, 'SK-MEL-30': 12, 'PC-3': 13, 'hTCEpi': 14, 
        #                        'HEK293': 15, 'HEL': 16, 'Hep-G2': 17, 'CACO-2': 18, 'OE19': 19, 'NIH 3T3': 20, 'JURKAT': 21, 'HaCaT': 22, 
        #                        'AF22': 23, 'SuSa': 24, 'THP-1': 25, 'REH': 26, 'EFO-21': 27, 'hTERT-RPE1': 28, 'ASC52telo': 29, 
        #                        'K-562': 30, 'HDLM-2': 31, 'HAP1': 32, 'HBEC3-KT': 33, 'NB4': 34, 'LHCN-M2': 35, 'PODO SVTERT152': 36, 
        #                        'PODO TERT256': 37, 'HUVEC TERT2': 38, 'RPTEC TERT1': 39}

        self.cell_line_dict = pickle.load(open(label_dict, "rb"))
        #self.annotation_dict = pickle.load(open(annotation_dict, "rb"))
        self.label_dict = pickle.load(open(annotation_dict, "rb"))
    
    def __len__(self):
        
        return len(self.flist)
    
    def __getitem__(self, idx):
        ret = {}
        file_name = str(self.flist[idx]).zfill(5)
        img = imread('{}/{}'.format(self.data_root, file_name))

        # img *= 2
        # img -= 1

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
        # assert cond_img_size[0] == 3, "Condition image should have 3 channels"
        # assert cond_img_size[1] == 512, "Condition image should have 256 height"
        # assert cond_img_size[2] == 512, "Condition image should have 256 width"

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
        #ret['annotation'] = torch.from_numpy(np.array(self.annotation_dict[ab][cell_line]))
        ret['path'] = file_name

        ret["clip_image"] = img_clip
        return ret