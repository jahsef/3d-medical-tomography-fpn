
STORE
global coords (kinda useless but), local_rs_coords, float local_ds_coords, ds_factor, ds_sigma, angstrom_sigma (angstrom sigma kinda useless here but its just nice metadata to have), voxel_spacing (same thing, just gonna keep it as metadata)
allows us to use shifts in realspace while keeping it all in downsampled space for gaussian
could allow us to use shear as well 

benefits:
half data (store patch and (1/16)^3 gaussian vs patch and gaussian), negligible data overhead for more flexibility 
we can choose to either use the precomputed cheap shit and we have the keypoint data for augmentations
augmentation accuracy retained
no interpolation artifacts, true gaussian reconstruction
memory efficiency (loads less data into memory of course)
faster compute (analytical shift operations recomputing gaussians vs scipy shifts)
even with multiple workers, monai is slow, especially if you didnt save gaussian at all and just the keypoints,
 if you generate full gaussians on the fly then you are almost forced to use gpu acceleration for gaussian ops which of course wastes gpu mem and compute

avoids monai's strict rules of same shape

color, brightness, noise, etc (use torch/monai/kornia)
scale, shift, shear, rotation (need to write custom ones because our gaussians are downsampled, shape mismatch with monai)
random crop (essentially just random shift but we need to mirror the stuff, should be fine with scipy shift or whatever)
coarse dropout (might be whatever) mixup, cutup, cutmix etc (hard to write on my own but might be worth?)
mixup: global mixing PROBABLY EASIEST OF THE HARD ONES TO DO
cutmix: rectangular paste
cutmixup: rectangular mixing


SE ABLATION, BCE (sanity check)

fpn ablation, bce (parallel, cascade, pc (fusion of both approaches))




fold 0 ablation for bce, combined (beta sweep), fuzzy (beta sweep)
inference on in fold across epochs and OOD data across epochs

beta scheduling as well

  1. ~~Combined more robust across sampling~~ (can't prove cleanly)
  2. Combined more robust to overfitting - you can show this with training curves
  3. Combined learns faster - show epoch to convergence
  4. Combined beats BCE on same config - your current data



WRITE THE CUSTOM AUGS PIPELINE

full 4 fold cv for winner (needs dropout and augs)
inference on in fold across epochs and OOD data across epochs
