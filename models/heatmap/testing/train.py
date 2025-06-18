import monai.transforms
import torch.nn.backends
import torchvision
# import torchvision.transforms as transforms
# import torchvision.transforms.v2 as t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torchvision.transforms
from torchvision.ops import sigmoid_focal_loss
from trainer import Trainer
import time
import os
import pandas as pd
from balancedrandomnsampler import BalancedRandomNSampler

from pathlib import Path
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs.motoridentifier import MotorIdentifier
from model_defs.trivialmodel import TrivialModel
from sklearn.model_selection import train_test_split

import utils
from patchtomodataset import PatchTomoDataset

from natsort import natsorted
import imageio.v3 as iio
import numpy as np

# import torchio as tio
import monai
from monai import transforms


def write_tomos(list_val_paths):
    """
    writes tomos if they dont exist, used for validation
    """
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    transform = t.Compose([
    t.ToDtype(torch.float16, scale=True),
    t.Normalize((0.479915,), (0.224932,))
    ])
    
    dst = Path.cwd() / 'normalized_val_fulltomo'
    for patches_path in list_val_paths:
        path:Path

        tomo_id = patches_path.name
        
        print(tomo_id)
        
        tomo_pt_path = dst / Path(str(tomo_id) + '.pt')
        
        if tomo_pt_path.exists():
            continue
        
        print(f'Writing full tomogram: {patches_path.name}')
        #find original images path
        images_path = Path.cwd() / 'original_data/train' / patches_path.name
        
        files = [
            f for f in images_path.rglob('*')
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]
        
        files = natsorted(files, key=lambda x: x.name)
        
        imgs = [iio.imread(file, mode="L") for file in files]
        
        tomo_array = np.stack(imgs)
        
        # Convert to tensor and normalize
        tomo_tensor = torch.as_tensor(tomo_array)
        tomo_tensor = transform(tomo_tensor)
        
        torch.save(tomo_tensor, tomo_pt_path)

import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()



    
if __name__ == "__main__":
    
    train_transform = None
    # intensity
    # rotation
    # scale
    # spatial
    
    train_transform = monai.transforms.Compose([
        #mild intensity
        transforms.RandGaussianNoised(keys = 'patch' ,dtype = torch.float16, prob = 0.5, std = 0.01),
        transforms.RandShiftIntensityd(keys = 'patch', offsets = 0.1,safe = True, prob = 0.50, ),
        
        #mild spatial/rotational
        # transforms.RandRotate90d(keys=["patch", "label"], prob=0.5),
        # transforms.SpatialPadd(keys = ['patch', 'label'], spatial_size= [80,80,80], mode = 'reflect'),
        # transforms.RandSpatialCropd(keys = ['patch', 'label'], roi_size = [64,64,64], random_center=True),
        
        #slightly more aggressive ones below
        # transforms.RandRotated(keys=["patch", "label"], range_x=0.45, range_y=0.45, range_z=0.45, prob=0.5),
        # transforms.RandZoomd(keys=["patch", "label"], min_zoom=0.8, max_zoom=1.2, prob=0.5),
        # transforms.RandGaussianSmoothd(keys="patch", sigma_x=(0.25, 0.35), sigma_y=(0.25, 0.35), sigma_z=(0.25, 0.35), prob=0.5),
        # transforms.RandAdjustContrastd(keys="patch", gamma=(0.8, 1.25), prob=0.5)
        
    ])
    
    train_transform = None
    
    
    #TODO visualization/logging
    #log f1 beta weighted stuff with precision + recall too
    #plot lr vs epoch
    #log some basic predictions + slices + ground_truth for a few key examples
    #maybe we can run predictions on a few tomos at the end of training then show ground truth vs prediction?

    #apply max, min, and average pooling to get some good visualizations
    #over depth dimension
    #of base image and convolutions??

    #plot predicted on a certain slice, also plot ground truth on the same slice or another one if needed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    max_motors = 1#max motors must be the same as our labels
    
    model = MotorIdentifier(max_motors=max_motors)
    
    # model = TrivialModel()
    
    print('loading state dict into model\n'*20)
    model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\heatmap\testing\best.pt'))
    
    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]
    # tomo_id_list = tomo_id_list[:len(tomo_id_list)]
    
    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size= 0.95, test_size= 0.05, random_state= 42)
    # train_id_list = train_id_list[:len(train_id_list)//10]
    train_id_list = ['tomo_d7475d']
    # print(f'train tomograms for debugging: {train_id_list}')
    # val_id_list = val_id_list[:len(val_id_list)//10]
    val_id_list = []
    
    #['tomo_bdc097', 'tomo_d7475d', 'tomo_51a47f', 'tomo_2c607f', 'tomo_975287', 'tomo_51a77e', 'tomo_3e7407', 'tomo_412d88', 'tomo_91beab', 'tomo_cc65a9', 'tomo_1f0e78', 'tomo_e71210', 'tomo_00e463', 'tomo_f36495', 'tomo_6943e6', 'tomo_711fad', 'tomo_aff073', 'tomo_fe050c', 'tomo_24795a', 'tomo_c46d3c', 'tomo_be4a3a', 'tomo_0d4c9e', 'tomo_821255', 'tomo_47ac94', 'tomo_ac4f0d', 'tomo_12f896', 'tomo_675583', 'tomo_20a9ed', 'tomo_b2b342', 'tomo_28f9c1', 'tomo_94c173', 'tomo_935f8a', 'tomo_746d88', 'tomo_8e4919', 'tomo_da79d8', 'tomo_40b215', 'tomo_c36b4b', 'tomo_1af88d', 'tomo_a2a928', 'tomo_13973d', 'tomo_c4db00', 'tomo_568537', 'tomo_101279', 'tomo_512f98', 'tomo_7fbc49', 'tomo_0333fa', 'tomo_f2fa4a', 'tomo_a37a5c', 'tomo_ec607b', 'tomo_a8bf76', 'tomo_dfc627', 'tomo_7a9b64', 'tomo_8b6795', 'tomo_23a8e8', 'tomo_651ecd', 'tomo_67565e', 'tomo_e9fa5f', 'tomo_2bb588', 'tomo_3a0914', 'tomo_10c564', 'tomo_8e30f5', 'tomo_b2eb0c', 'tomo_16fce8', 'tomo_fe85f6', 'tomo_f672c0', 'tomo_67ff4e', 'tomo_38d285', 'tomo_d23087', 'tomo_2b996c', 'tomo_098751', 'tomo_2e1f4c', 'tomo_369cce', 'tomo_7eb641', 'tomo_d3bef7', 'tomo_225d8f', 'tomo_bdd3a0', 'tomo_41ea80', 'tomo_1fb6a7', 'tomo_e61cdf', 'tomo_1cc887', 'tomo_d2339b', 'tomo_cae587', 'tomo_8e58f1', 'tomo_35ec84', 'tomo_183270', 'tomo_532d49', 'tomo_e2ccab', 'tomo_85708b', 'tomo_ea3f3a', 'tomo_f3e449', 'tomo_372690', 'tomo_53e048', 'tomo_ca8be0', 'tomo_974fd4', 'tomo_b24f1a', 'tomo_61e947', 'tomo_eb26ee', 'tomo_8554af', 'tomo_0c3d78', 'tomo_7550f4', 'tomo_6e196d', 'tomo_d96d6e', 'tomo_7036ee', 'tomo_dee783', 'tomo_e32b81', 'tomo_b0e5c6', 'tomo_518a1f', 'tomo_6f83d4', 'tomo_7f0184', 'tomo_84997e', 'tomo_4e38b8', 'tomo_ecbc12', 'tomo_b18127', 'tomo_98d455', 'tomo_76a42b', 'tomo_5f235a', 'tomo_d2b1bc', 'tomo_2a89bb', 'tomo_2cace2', 'tomo_3b83c7', 'tomo_57592d', 'tomo_033ebe', 'tomo_6303f0', 'tomo_bad724', 'tomo_85fa87', 'tomo_b7d94c', 'tomo_823bc7', 'tomo_7fa3b1', 'tomo_6f2c1f', 'tomo_37c426', 'tomo_539259', 'tomo_122c46', 'tomo_122a02', 'tomo_1dc5f9', 'tomo_c36baf', 'tomo_f0adfc', 'tomo_1b82d1', 'tomo_049310', 'tomo_4f5a7b', 'tomo_cf53d0', 'tomo_d84544', 'tomo_acadd7', 'tomo_656915', 'tomo_e77217', 'tomo_d0c025', 'tomo_e50f04', 'tomo_226cd8', 'tomo_9dbc12', 'tomo_7b1ee3', 'tomo_9fc2b6', 'tomo_210371', 'tomo_08bf73', 'tomo_5b34b2', 'tomo_8634ee', 'tomo_95e699', 'tomo_5e2a91', 'tomo_5d798e', 'tomo_7a49bd', 'tomo_651ec2', 'tomo_374ca7', 'tomo_6a84b7', 'tomo_5dd63d', 'tomo_e2da77', 'tomo_5308e8', 'tomo_0363f2', 'tomo_efe1f8', 'tomo_738500', 'tomo_8ee8fd', 'tomo_fb08b5', 'tomo_e1a034', 'tomo_22976c', 'tomo_72b187', 'tomo_b11ddc', 'tomo_510f4e', 'tomo_4ed9de', 'tomo_6bc974', 'tomo_24fda8', 'tomo_cff77a', 'tomo_c11e12', 'tomo_e685b8', 'tomo_0c3a99', 'tomo_49f4ee', 'tomo_cc3fc4', 'tomo_47d380', 'tomo_6607ec', 'tomo_3b7a22', 'tomo_1efc28', 'tomo_8f4d60', 'tomo_24a095', 'tomo_8e8368', 'tomo_1c38fd', 'tomo_97a2c6', 'tomo_85edfd', 'tomo_4c1ca8', 'tomo_05f919', 'tomo_ff505c', 'tomo_a4c52f', 'tomo_23ce49', 'tomo_d634b7', 'tomo_2dd6bd', 'tomo_fc1665', 'tomo_bfd5ea', 'tomo_0e9757', 'tomo_003acc', 'tomo_fd41c4', 'tomo_63e635', 'tomo_a549d6', 'tomo_9f1828', 'tomo_e34af8', 'tomo_806a8f', 'tomo_08a6d6', 'tomo_191bcb', 'tomo_2dcd5c', 'tomo_e764a7', 'tomo_5b359d', 'tomo_247826', 'tomo_49725c', 'tomo_6acb9e', 'tomo_6d22d1', 'tomo_072a16', 'tomo_774aae', 'tomo_9986f0', 'tomo_38c2a6', 'tomo_95c0eb', 'tomo_82d780', 'tomo_e51e5e', 'tomo_ac9fef', 'tomo_a2bf30', 'tomo_134bb0', 'tomo_66285d', 'tomo_4e478f', 'tomo_ff7c20', 'tomo_d396b5', 'tomo_2b3cdf', 'tomo_e3864f', 'tomo_b7d014', 'tomo_881d84', 'tomo_bb9df3', 'tomo_fa5d78', 'tomo_e2a336', 'tomo_bd42fa', 'tomo_bebadf', 'tomo_a8073d', 'tomo_3a3519', 'tomo_30b580', 'tomo_ca472a', 'tomo_dae195', 'tomo_4b124b', 'tomo_fb6ce6', 'tomo_bf1398', 'tomo_fc5ae4', 'tomo_e7c195', 'tomo_381add', 'tomo_37dd38', 'tomo_cacb75', 'tomo_081a2d', 'tomo_455dcd', 'tomo_cf0875', 'tomo_04d42b', 'tomo_0a180f', 'tomo_b4a1f0', 'tomo_444829', 'tomo_a67e9f', 'tomo_8e90f9', 'tomo_622ca9', 'tomo_d8c917', 'tomo_69d7c9', 'tomo_3b8291', 'tomo_317656', 'tomo_a4f419', 'tomo_6e237a', 'tomo_2c9f35', 'tomo_b33d4e', 'tomo_46250a', 'tomo_17143f', 'tomo_9f918e', 'tomo_5764d6', 'tomo_6c4df3', 'tomo_06e11e', 'tomo_8f063a', 'tomo_648adf', 'tomo_935ae0', 'tomo_f6a38a', 'tomo_4d528f', 'tomo_997240', 'tomo_df866a', 'tomo_8174f5', 'tomo_256717', 'tomo_4925ee', 'tomo_754447', 'tomo_bde7f3', 'tomo_aec312', 'tomo_d9a2af', 'tomo_c00ab5', 'tomo_0308c5', 'tomo_13484c', 'tomo_ab30af', 'tomo_b9de3e', 'tomo_c596be', 'tomo_066095', 'tomo_138018', 'tomo_f82a15', 'tomo_e6f7f7', 'tomo_01a877', 'tomo_e57baf', 'tomo_a5ac23', 'tomo_7dcfb8', 'tomo_6cf2df', 'tomo_adc026', 'tomo_6df2d6', 'tomo_918e2b', 'tomo_59b470', 'tomo_0a8f05', 'tomo_c8f3ce', 'tomo_6733fa', 'tomo_139d9e', 'tomo_5b8db4', 'tomo_0fab19', 'tomo_3e6ead', 'tomo_5f1f0c', 'tomo_f6de9b', 'tomo_466489', 'tomo_d662b0', 'tomo_b4d92b', 'tomo_4e41c2', 'tomo_d6e3c7', 'tomo_9aee96', 'tomo_f7f28b', 'tomo_d916dc', 'tomo_1c2534', 'tomo_997437', 'tomo_423d52', 'tomo_e5ac94', 'tomo_4077d8', 'tomo_79d622', 'tomo_4e3e37', 'tomo_f8b46e', 'tomo_93c0b4', 'tomo_abb45a', 'tomo_3b1cc9', 'tomo_9cd09e', 'tomo_aaa1fd', 'tomo_5b087f', 'tomo_9f222a', 'tomo_cb5ec6', 'tomo_2a6ca2', 'tomo_ba76d8', 'tomo_1e9980', 'tomo_c7b008', 'tomo_f78e91', 'tomo_2f3261', 'tomo_b98cf6', 'tomo_f8b835', 'tomo_ede779', 'tomo_517f70', 'tomo_d26fcb', 'tomo_a910fe', 'tomo_a46b26', 'tomo_d0699e', 'tomo_03437b', 'tomo_b50c0f', 'tomo_0f9df0', 'tomo_378f43', 'tomo_646049', 'tomo_769126', 'tomo_cad74b', 'tomo_ba37ec', 'tomo_957567', 'tomo_180bfd', 'tomo_2c8ea2', 'tomo_a9d067', 'tomo_ae347a', 'tomo_ab78d0', 'tomo_672101', 'tomo_d5465a', 'tomo_73173f', 'tomo_3a6a9d', 'tomo_676744', 'tomo_db4517', 'tomo_91031e', 'tomo_e0739f', 'tomo_0fe63f', 'tomo_3183d2', 'tomo_a75c98', 'tomo_79756f', 'tomo_4f379f', 'tomo_8e4f7d', 'tomo_ab804d', 'tomo_399bd9', 'tomo_e96200', 'tomo_f76529', 'tomo_a84050', 'tomo_e1e5d3', 'tomo_39b15b', 'tomo_c6f50a', 'tomo_8d5995', 'tomo_81445c', 'tomo_ba9b3d', 'tomo_307f33', 'tomo_d6c63f', 'tomo_2645a0', 'tomo_385eb6', 'tomo_5f34b3', 'tomo_9ed470', 'tomo_b4d9da', 'tomo_278194', 'tomo_9674bf', 'tomo_813916', 'tomo_6c203d', 'tomo_71d2c0', 'tomo_6478e5', 'tomo_a020d7', 'tomo_305c97', 'tomo_499ee0', 'tomo_50f0bf', 'tomo_b03f81', 'tomo_80bf0f', 'tomo_62dbea', 'tomo_a72a52', 'tomo_c649f8', 'tomo_bfdf19', 'tomo_d31c96', 'tomo_f07244', 'tomo_dfdc32', 'tomo_fc90fd', 'tomo_30d4e5', 'tomo_a1a9a3', 'tomo_c9d07c', 'tomo_c38e83', 'tomo_57c814', 'tomo_89d156', 'tomo_16136a', 'tomo_569981', 'tomo_3264bc', 'tomo_e5a091', 'tomo_cabaa0', 'tomo_4b59a2', 'tomo_e63ab4', 'tomo_b0ded6', 'tomo_f427b3', 'tomo_288d4f', 'tomo_be9b98', 'tomo_9a7701', 'tomo_f871ad', 'tomo_71ece1', 'tomo_67717e', 'tomo_1446aa', 'tomo_b10aa4', 'tomo_c4a4bb', 'tomo_2c9da1', 'tomo_48dc93', 'tomo_ef1a1a', 'tomo_db2a10', 'tomo_d0aa3b', 'tomo_cd1a7c', 'tomo_9722d1', 'tomo_7cf523', 'tomo_b9088c', 'tomo_3a8480', 'tomo_4555b6', 'tomo_5d4c65', 'tomo_78b03d', 'tomo_ca1d13', 'tomo_4102f1', 'tomo_7e3494', 'tomo_372a5c', 'tomo_a537dd', 'tomo_e26c6b', 'tomo_4ee35e', 'tomo_8f5995', 'tomo_6521dc', 'tomo_5d01e8', 'tomo_d0d9b6', 'tomo_2daaee', 'tomo_2fb12d', 'tomo_b28579', 'tomo_6cb0f0', 'tomo_97876d', 'tomo_9997b3', 'tomo_5984bf', 'tomo_fd5b38', 'tomo_cf5bfc', 'tomo_bb5ac1', 'tomo_8d2d48', 'tomo_c7a40f', 'tomo_fc3c39', 'tomo_e81143', 'tomo_d723cd', 'tomo_229f0a', 'tomo_60ddbd', 'tomo_3e7783', 'tomo_0eb994', 'tomo_4469a7', 'tomo_d83ff4', 'tomo_d56709', 'tomo_ed1c97', 'tomo_25780f', 'tomo_603e40', 'tomo_b87c8e', 'tomo_dc9a96', 'tomo_516cdd', 'tomo_9d3a0e', 'tomo_9f424e', 'tomo_d5aa20', 'tomo_556257', 'tomo_02862f', 'tomo_e22370', 'tomo_285454', 'tomo_a3ed10', 'tomo_c678d9', 'tomo_136c8d', 'tomo_0de3ee', 'tomo_507b7a', 'tomo_37076e', 'tomo_0c2749', 'tomo_e55f81', 'tomo_5ba0cf', 'tomo_c10f64', 'tomo_983fce', 'tomo_6f0ee4', 'tomo_a81e01', 'tomo_9cde9d', 'tomo_c4bfe2', 'tomo_375513', 'tomo_19a313', 'tomo_bad7b3', 'tomo_319f79', 'tomo_f94504', 'tomo_cc2b5c', 'tomo_1ab322', 'tomo_05df8a', 'tomo_3c6038', 'tomo_10a3bd', 'tomo_c3619a', 'tomo_98686a', 'tomo_4baff0', 'tomo_54e1a7', 'tomo_146de2', 'tomo_868255', 'tomo_72763e', 'tomo_640a74', 'tomo_b8f096', 'tomo_b7becf', 'tomo_fea6e8', 'tomo_68e123', 'tomo_50cbd9', 'tomo_401341', 'tomo_da38ea', 'tomo_a0cb00', 'tomo_bcb115', 'tomo_2acf68', 'tomo_dd36c9', 'tomo_9c0253', 'tomo_00e047', 'tomo_eb4fd4', 'tomo_db656f', 'tomo_221a47', 'tomo_513010', 'tomo_0eb41e', 'tomo_c84b8e', 'tomo_abac2e', 'tomo_b54396', 'tomo_94a841', 'tomo_2483bb', 'tomo_88af60', 'tomo_dcb9b4', 'tomo_8c13d9', 'tomo_60d478', 'tomo_05b39c', 'tomo_7dc063', 'tomo_e9b7f2', 'tomo_9ae65f', 'tomo_6a6a3b', 'tomo_417e5f', 'tomo_fd9357', 'tomo_bede89', 'tomo_fadbe2', 'tomo_c77de0', 'tomo_616f0b', 'tomo_decb81', 'tomo_dbc66d', 'tomo_4c2e4e', 'tomo_bc143f', 'tomo_c925ee', 'tomo_172f08', 'tomo_bbe766', 'tomo_db6051', 'tomo_643b20', 'tomo_087d64', 'tomo_7ca7c0', 'tomo_b8595d', 'tomo_3eb9c8', 'tomo_6b1fd3', 'tomo_4e1b18', 'tomo_99a3ce', 'tomo_a6646f', 'tomo_c13fbf', 'tomo_8acc4b', 'tomo_79a385', 'tomo_32aaa7', 'tomo_285d15', 'tomo_971966', 'tomo_23c8a4', 'tomo_b80310', 'tomo_8351d1', 'tomo_53c71b', 'tomo_b9eb9a', 'tomo_2fc82d', 'tomo_f1bf2f', 'tomo_08446f', 'tomo_1da097', 'tomo_2aeb29', 'tomo_692081', 'tomo_aeaf51', 'tomo_2a6091']
    # print(f'validation tomograms for debugging: {val_id_list}')
    #['tomo_fbb49b', 'tomo_56b9a3', 'tomo_e72e60', 'tomo_abbd3b', 'tomo_1da0da', 'tomo_b2ebbc', 'tomo_493bea', 'tomo_331130', 'tomo_5bb31c', 'tomo_6bb452', 'tomo_ec1314', 'tomo_6c5a26', 'tomo_634b06', 'tomo_221c8e', 'tomo_b93a2d', 'tomo_62eea8', 'tomo_91c84c', 'tomo_e8db69', 'tomo_1c75ac', 'tomo_040b80', 'tomo_891730', 'tomo_19a4fd', 'tomo_891afe', 'tomo_161683', 'tomo_d21396', 'tomo_8d231b', 'tomo_47c399', 'tomo_736dfa', 'tomo_c84b46', 'tomo_0da370', 'tomo_f71c16', 'tomo_16efa8', 'tomo_464108']
    
    batch_size = 64
    batches_per_step = 2 #for gradient accumulation (every n batches we step)
    steps_per_epoch = 4
    std_dev = 6

    train_dataset = PatchTomoDataset(
        sigma=std_dev,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        transform = train_transform,
        tomo_id_list= train_id_list
    )
    
    val_dataset = PatchTomoDataset(
        sigma=std_dev,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        transform = None,
        tomo_id_list= val_id_list
    )
    
    pin_memory = True
    num_workers = 3
    persistent_workers = True
    prefetch_factor = 1
    
    sampler = BalancedRandomNSampler(train_dataset, n = batch_size*batches_per_step*steps_per_epoch, balance_ratio= 0.1, class_labels= train_dataset.index_df['has_motor'])
    
    train_loader = DataLoader(train_dataset,sampler = sampler, batch_size = batch_size, shuffle = False, pin_memory =pin_memory, num_workers=num_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)

    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory =pin_memory, num_workers=num_workers//2, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)
    
    #regression loss with all false labels results in nan
    #setting regression loss to 0 in that case
    # regression_loss_fn = torch.nn.SmoothL1Loss(beta = 1)#lower beta = more robust to outliers
    
    # pos_weight = torch.tensor([10]).to(device)
    # conf_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # conf_loss_fn = torch.nn.MSELoss()
    
    conf_loss_fn = FocalLoss(alpha = 20, gamma = 2)#alpha is pos weight, gamma is focusing param
    
    epochs = 100
    print(f'TOTAL EXPECTED PATCHES TRAINED: {batch_size*batches_per_step*steps_per_epoch*epochs}')
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    
    warmup_steps = int(0.05 * total_steps)#% of steps warmup, 5% is about 2 epochs
    
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        import math
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress)) + 0.01  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-6, weight_decay= 1e-4)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps= warmup_steps, total_steps= total_steps)
    
    save_dir = './models/heatmap/testing/'

    os.makedirs(save_dir, exist_ok= True)


    # Train and validate the model
    trainer = Trainer(
        model=model,
        batches_per_step = batches_per_step,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler = scheduler,
        # regression_loss_fn=regression_loss_fn,
        conf_loss_fn = conf_loss_fn,
        # regression_loss_weight = 1.0,
        # conf_loss_weight= 2.0,
        device=device,
        save_dir = save_dir
        )
    
    trainer.train(
        epochs=epochs,
        save_period=0
    )
    