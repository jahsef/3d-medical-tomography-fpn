super interestingly 

    OLD focal 3 pos 1.2 gamma (targets - pt) ; pt = exp(-bce), instability perhaps from saturated logits?
epoch,train_conf_loss,train_dice,train_comp,val_conf_loss,val_dice,val_comp,train_top10,train_top50,train_top300,val_top10,val_top50,val_top300
25,0.008957,0.350357,0.75049,0.0,0.0,0.0,0.850444,0.903676,0.972117,0.0,0.0,0.0

Best F-2 score: 0.7660
Best threshold: 0.85
Precision: 0.9500, Recall: 0.7310

    NEW FOCAL 3 pos 1.2 gamma (targets - pred_prob); pred_prob = sigmoid(inputs)
epoch,train_conf_loss,train_dice,train_comp,val_conf_loss,val_dice,val_comp,train_top10,train_top50,train_top300,val_top10,val_top50,val_top300
22,0.002562,0.264226,0.731057,0.0,0.0,0.0,0.826362,0.883397,0.974651,0.0,0.0,0.0
23,0.002259,0.276831,0.745175,0.0,0.0,0.0,0.820025,0.864385,0.945501,0.0,0.0,0.0
24,0.002388,0.281722,0.731111,0.0,0.0,0.0,0.833967,0.904943,0.984791,0.0,0.0,0.0
25,0.00237,0.288384,0.743328,0.0,0.0,0.0,0.798479,0.841572,0.940431,0.0,0.0,0.0
26,0.0021,0.2926,0.745065,0.0,0.0,0.0,0.808619,0.86692,0.951838,0.0,0.0,0.0

Best F-2 score: 0.6970
Best threshold: 0.73
Precision: 0.9440, Recall: 0.6540

    bce (epoch 25 for direct comparison, 30 is what my training got to before pc died)
epoch,train_conf_loss,train_dice,train_comp,val_conf_loss,val_dice,val_comp,train_top10,train_top50,train_top300,val_top10,val_top50,val_top300
25,0.013745,0.312061,0.723017,0.0,0.0,0.0,0.797212,0.856781,0.946768,0.0,0.0,0.0
30,0.012997,0.33664,0.746667,0.0,0.0,0.0,0.856781,0.887199,0.95057,0.0,0.0,0.0

Best F-2 score: 0.6930
Best threshold: 0.73
Precision: 0.5760, Recall: 0.7310

so our custom focal loss truely seems to be better (slightly better metrics but way better F-Beta)
just need to fix up instability issues (trying more aggressive grad clipping 100 => 6.7)



actual ablation

vanilla bce (divergence after epoch 59 on top-k metrics due to overfitting regime, but in full tomogram validation of the same training set, the divergent ones seemed to be better??)
epoch,train_conf_loss,train_dice,train_peak_dist,train_peak_sharp,val_conf_loss,val_dice,val_peak_dist,val_peak_sharp,train_top10,train_top50,train_top300,val_top10,val_top50,val_top300
45,0.013961,0.320625,0.849922,0.80898,0.0,0.0,0.0,0.0,0.787072,0.854246,0.929024,0.0,0.0,0.0
59,0.011756,0.356201,0.84925,0.812746,0.0,0.0,0.0,0.0,0.910013,0.964512,0.992395,0.0,0.0,0.0
99,0.011201,0.367727,0.887305,0.798853,0.0,0.0,0.0,0.0,0.776933,0.78327,0.865653,0.0,0.0,0.0

epoch 45
Best F-2 score: 0.6000
Best threshold: 0.87
Precision: 0.7140, Recall: 0.5770
epoch 59
Best F-2 score: 0.6900
Best threshold: 0.79
Precision: 0.4880, Recall: 0.7690
epoch 99
Best F-2 score: 0.7720
Best threshold: 0.83
Precision: 0.6560, Recall: 0.8080

epochs to milestone
milestone top10  top50 top300
0.6         30     22     5
0.8         47     37     20
0.85        56     42     29
0.9         58     49     38
0.92        NAN    56     42
0.95        NAN    57     49
0.96        NAN    NAN    49
0.99        NAN    NAN    58
1.00        NAN    NAN    NAN



new focal


Best F-2 scores by prune radius:
  r=0: F2=0.7360 @ thresh=0.89 (P=0.407, R=0.923)
  r=1: F2=0.8080 @ thresh=0.89 (P=0.808, R=0.808)
  r=2: F2=0.8080 @ thresh=0.89 (P=0.808, R=0.808)
  r=3: F2=0.8080 @ thresh=0.89 (P=0.808, R=0.808)
  r=4: F2=0.8080 @ thresh=0.89 (P=0.808, R=0.808)
  r=5: F2=0.8080 @ thresh=0.89 (P=0.808, R=0.808)

epochs to milestone
milestone top10  top50 top300
0.6         18     15     2
0.8         25     20     13
0.85        29     23     16
0.9         
0.92   
0.95   
0.96  
0.99     
1.00   