Empirical loss function study for 3D sparse detection:

MSE vs Focal vs CornerNet in 3D cryo-EM
Continuous focal adaptation (obvious but undocumented)
β scheduling curriculum (+3-5pp improvement)




Architecture insights:

Parallel FPN > Cascade FPN for sparse detection 
both are 25 epochs of training with early stopping
parallel top 10, 50, 300 (3240 total voxels)
0.963245,0.977186,0.984791
cascade
0.716096,0.732573,0.7782

cascade performing worse does not have a concrete explanation from us and warrants further investigation
similar param counts
not gradient flow (adding skips didnt help)
not feature confusion (no skips didnt work either)


using relabeled data found at
https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset/data?select=labels_new.csv
original competition labels found at
https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025

Relabeling (+33% motors found)
Better sampling (multi-motor, hard negatives)
Pipeline efficiency


Efficiency comparison:

Competitive with 3.8× less data (our 648 original tomograms vs 2000+ tomograms winners used), 40× fewer parameters (50m param model with ensemble and TTA)