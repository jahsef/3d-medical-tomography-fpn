import monai.inferers
import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d


from .. import nnblock
from ..nnblock import check_tensor
    
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import monai
import gc
import time
import torch.nn.functional as F
import math

from .utils import dc_growth_fn as _dc_growth_fn, power_2_rounding, growth_fn as _growth_fn, raw_growth_fn as _raw_growth_fn

#PC FPN: parallel cascade fpn, uses heavily aggregated heavily compressed sequential processing (cascade) + aggregated mildly compressed parallel processing (parallel portion) into head
#so intuitively we have cascade neck and parallel neck here

class PCFPN(nn.Module):

    def __init__(self, base_features, growth_rate, downchanneling_factor, dropout_p=0, drop_path_p=0):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_path_p = drop_path_p
        F_I = base_features
        G_R = growth_rate
        DC = downchanneling_factor

        # Local wrappers that capture F_I, G_R, DC
        def gf(level):
            return _growth_fn(level, F_I, G_R)

        def dcf(level):
            return _dc_growth_fn(level, F_I, G_R, DC)

        def rgf(level):
            return _raw_growth_fn(level, F_I, G_R)

        def cascade_in(level):
            # assert level >= 3, 'cascade in must be >= 3'
            if level == 3:
                return dcf(3)
            else:
                return power_2_rounding(dcf(level-1)/G_R) + dcf(level)

        def cascade_out(level):
            # assert level >= 3, 'cascade out must be >= 3'
            return power_2_rounding(dcf(level)/G_R)


        def hgf(level, c,pndc):
            # head input is cascade_out(5) + all DOWNCHANNELED skip connections
            head_input = cascade_out(5) + power_2_rounding(c/(pndc**2))
            return head_input if level == 5 else power_2_rounding(head_input/(DC**(5-level)))
        
        
        self.stem = nn.Conv3d(1, F_I, kernel_size=5, stride=1, padding=2)

        self.enc_2 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, gf(1), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(1), gf(1), kernel_size=3, norm_type="gn", drop_path_p=drop_path_p)
        )
        self.enc_4 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(1), gf(2), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(2), gf(2), kernel_size=3, norm_type="gn", drop_path_p=drop_path_p)
        )
        self.enc_8 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(2), gf(3), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
        )
        self.enc_16 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(3), gf(4), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
        )
        self.enc_32 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(4), gf(5), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
        )

        #this architecture only uses 1/8, 1/16, 1/32 scale for cascade and parallel neck
        self.interp_r_8 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(3), dcf(3), drop_path_p=drop_path_p, norm_type="gn"),
            # nnblock.get_norm_layer("gn", dcf(3)),
            nnblock.SEBlock3d(dcf(3), reduction=4),
            nnblock.get_norm_layer("gn", dcf(3)),
        )
        self.interp_r_16 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(4), dcf(4), drop_path_p=drop_path_p, norm_type="gn"),
            # nnblock.get_norm_layer("gn", dcf(4)),
            nnblock.SEBlock3d(dcf(4), reduction=4),
            nnblock.get_norm_layer("gn", dcf(4)),
        )
        self.interp_r_32 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(5), dcf(5), drop_path_p=drop_path_p, norm_type="gn"),
            # nnblock.get_norm_layer("gn", dcf(5)),
            nnblock.SEBlock3d(dcf(5), reduction=4),
            nnblock.get_norm_layer("gn", dcf(5)),
        )
        #on our diagram this cascade processing block is the same level as the number in its name
        #usage: takes in current level interp'd output concat with previous level output

        #CASCADE NECK (levels 8, 16, 32 only)
        self.cascade_8 = nn.Sequential(
            nnblock.PreActDownchannel3d(cascade_in(3), cascade_out(3),norm_type="gn",drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(cascade_out(3),cascade_out(3), drop_path_p=drop_path_p, norm_type="gn"),
            nnblock.get_norm_layer(norm_type="gn", num_channels=cascade_out(3)),
        )
        self.cascade_16 = nn.Sequential(
            nnblock.PreActDownchannel3d(cascade_in(4), cascade_out(4),norm_type="gn",drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(cascade_out(4),cascade_out(4), drop_path_p=drop_path_p, norm_type="gn"),
            nnblock.get_norm_layer(norm_type="gn", num_channels=cascade_out(4)),
        )
        self.cascade_32 = nn.Sequential(
            nnblock.PreActDownchannel3d(cascade_in(5), cascade_out(5), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(cascade_out(5),cascade_out(5), drop_path_p=drop_path_p, norm_type="gn"),
            nnblock.get_norm_layer(norm_type="gn", num_channels=cascade_out(5)),
        )

        raw_feature_progression = [rgf(i) for i in [1, 2, 3, 4, 5]]
        print(f'raw feature progression in backbone: {F_I}, {raw_feature_progression}')

        feature_progression = [gf(i) for i in [1, 2, 3, 4, 5]]
        print(f'feature progression in backbone: {F_I}, {feature_progression}')

        downchannel_progression = [dcf(i) for i in [3, 4, 5]]
        print(f'after interp downchanneling : {downchannel_progression}')

        # cascade channel progression (lean)
        cascade_progression = [(cascade_in(i), cascade_out(i)) for i in range(3, 6)]
        print(f'cascade progression (in->out): {cascade_progression}')
        parallel_neck_input = sum(downchannel_progression)
        PNDC = 1.5
        c = power_2_rounding(parallel_neck_input/3)
        print(f'parallel neck progression: input: {parallel_neck_input}, initial downchannel: {c}, {[power_2_rounding(c/(PNDC**n)) for n in [1,2]]}')

        

        print(f'head progression: {[hgf(i, c, pndc=PNDC) for i in [5, 4, 3, 2]]}, 1')
        
       
        self.parallel_neck = nn.Sequential(
            #so while this is technically double SE (outputs from interp are SE'd)
            #we global SE those 'local' SE from the interp blocks
            #might be suboptimal but im guessing the 'local' SE blocks will still help our cascade features so
            nn.SiLU(inplace=True),
            nnblock.SEBlock3d(parallel_neck_input, reduction=4),#
            nn.Conv3d(parallel_neck_input, c,kernel_size=1, bias=False),
            nnblock.PreActResBlock3d(c,power_2_rounding(c/(PNDC**1)), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(power_2_rounding(c/(PNDC**1)),power_2_rounding(c/(PNDC**2)), norm_type="gn", drop_path_p=drop_path_p),
            # nnblock.PreActResBlock3d(power_2_rounding(c/(PNDC**2)),power_2_rounding(c/(PNDC**3)), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type="gn", num_channels=power_2_rounding(c/(PNDC**2)))
        )
        
        # Organized module lists
        self.backbone = nn.ModuleList([self.stem, self.enc_2, self.enc_4, self.enc_8, self.enc_16, self.enc_32])
        self.neck = nn.ModuleList([
            self.interp_r_8, self.interp_r_16, self.interp_r_32,
            self.cascade_8, self.cascade_16, self.cascade_32
        ])


        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nn.SiLU(inplace=True),#act before se is unconventional but it seemed to show same or slightly better performance
            nnblock.SEBlock3d(hgf(5, c, pndc=PNDC), reduction=4),
            nn.Conv3d(hgf(5, c, pndc=PNDC), hgf(4, c, pndc=PNDC), kernel_size=1, bias=False),
            nnblock.PreActResBlock3d(hgf(4, c, pndc=PNDC), hgf(3, c, pndc=PNDC), kernel_size=3, drop_path_p=drop_path_p,norm_type="gn"),
            nnblock.PreActResBlock3d(hgf(3, c, pndc=PNDC), hgf(2, c, pndc=PNDC), kernel_size=3, drop_path_p=drop_path_p,norm_type="gn"),
            nnblock.get_norm_layer("gn", hgf(2, c, pndc=PNDC)),
            nn.SiLU(inplace=True),
            nn.Conv3d(hgf(2, c, pndc=PNDC), 1, kernel_size=1, bias = True) #MAKE SURE FINAL OUTPUT ISNT A SKIP CONNECTION!!!!!!!!!
        )

    def forward(self, x):
        assert not torch.isnan(x).any(), 'x nan yo'
        
        stem_x = self.stem(x)
        
        enc_2_x = self.enc_2(stem_x)
        enc_4_x = self.enc_4(enc_2_x)
        enc_8_x = self.enc_8(enc_4_x)
        enc_16_x = self.enc_16(enc_8_x)
        enc_32_x = self.enc_32(enc_16_x)

        target_size = enc_16_x.shape[2:]

        interp_8_x = F.interpolate(enc_8_x, size=target_size, mode='trilinear')
        
        interp_8_x = self.interp_r_8(interp_8_x)
        interp_16_x = F.interpolate(enc_16_x, size=target_size, mode='trilinear')
        
        interp_16_x = self.interp_r_16(interp_16_x) #we still interp 1/16th x still because the 1/16th 
        interp_32_x = F.interpolate(enc_32_x, size=target_size, mode='trilinear')
        
        interp_32_x = self.interp_r_32(interp_32_x)

        check_tensor('interp_8_x', interp_8_x)
        check_tensor('interp_16_x', interp_16_x)
        check_tensor('interp_32_x', interp_32_x)

        cascade8_x = self.cascade_8(interp_8_x)
        cascade16_x = self.cascade_16(torch.cat([cascade8_x, interp_16_x], dim=1))
        cascade32_x = self.cascade_32(torch.cat([cascade16_x, interp_32_x], dim=1))


        interp_x = torch.cat([interp_8_x, interp_16_x, interp_32_x], dim=1)
        parallel_neck_x = self.parallel_neck(interp_x)
        final_head_input = torch.cat([cascade32_x, parallel_neck_x], dim=1)    
        
        results = self.head(final_head_input)
        
        check_tensor("results", results, related_tensors={
            "interp_8_x": interp_8_x,
            "interp_16_x": interp_16_x,
            "interp_32_x": interp_32_x,
            "cascade32_x": cascade32_x,
        })
        
        return results

    def print_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Backbone params (stem + encoders)
        backbone_modules = [self.stem, self.enc_2, self.enc_4, self.enc_8, self.enc_16, self.enc_32]
        backbone_names = ['stem', 'enc_2', 'enc_4', 'enc_8', 'enc_16', 'enc_32']
        backbone_params = [sum(p.numel() for p in module.parameters() if p.requires_grad)
                          for module in backbone_modules]
        total_backbone = sum(backbone_params)

        # Interp blocks params
        interp_modules = [self.interp_r_8, self.interp_r_16, self.interp_r_32]
        interp_names = ['interp_r_8', 'interp_r_16', 'interp_r_32']
        interp_params = [sum(p.numel() for p in module.parameters() if p.requires_grad)
                        for module in interp_modules]
        total_interp = sum(interp_params)

        # Cascade neck params
        cascade_modules = [self.cascade_8, self.cascade_16, self.cascade_32]
        cascade_names = ['cascade_8', 'cascade_16', 'cascade_32']
        cascade_params = [sum(p.numel() for p in module.parameters() if p.requires_grad)
                         for module in cascade_modules]
        total_cascade = sum(cascade_params)

        # Parallel neck params
        parallel_neck_params = sum(p.numel() for p in self.parallel_neck.parameters() if p.requires_grad)

        # Head params
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        # Format details
        backbone_details = ', '.join(f'{name}: {params:,}' for name, params in zip(backbone_names, backbone_params))
        interp_details = ', '.join(f'{name}: {params:,}' for name, params in zip(interp_names, interp_params))
        cascade_details = ', '.join(f'{name}: {params:,}' for name, params in zip(cascade_names, cascade_params))

        print(
            f'Total trainable params: {trainable_params:,}\n'
            f'---\n'
            f'Backbone params: {total_backbone:,}\n'
            f'  {backbone_details}\n'
            f'---\n'
            f'Interp blocks params: {total_interp:,}\n'
            f'  {interp_details}\n'
            f'---\n'
            f'Cascade neck params: {total_cascade:,}\n'
            f'  {cascade_details}\n'
            f'---\n'
            f'Parallel neck params: {parallel_neck_params:,}\n'
            f'---\n'
            f'Head params: {head_params:,}'
        )




if __name__ == '__main__':
    model = PCFPN(4, 2.3, 2)
    model.print_params()