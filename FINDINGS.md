Empirical loss function study for 3D sparse detection:

MSE vs Focal vs CornerNet in 3D cryo-EM
Continuous focal adaptation (obvious but undocumented)
β scheduling curriculum (+3-5pp improvement)


SE BLOCKS ARE STILL HELPFUL EVEN THOUGH THEIR CHANNEL WEIGHTINGS ARE NEAR IDENTITY
THEORY:UNCLEAR
near identity weighting (sigmoid(4.0) = 0.98) for all channels, no deviation depending on sample
epochs to top10 0.8 is reduced by 40% (14 to 10)
epochs to top10 0.96 reduced by 75%   (60 to 15)

LOSS FUNCTION COMPARISON (full tomo inference, not patch metrics)

fuzzy α=2,β=4: 0.918 F2 @ epoch15 → 0.926 F2 @ epoch28 (best.pt)
  - only 2 checkpoints so degradation curve unclear
  - high threshold (0.63→0.37) suggests confident peaks

combined α=2,β=6: 0.900 @ epoch15 → 0.926 @ epoch25 → 0.784 @ epoch35 → 0.769 @ epoch45
  - matches fuzzy peak F2, degrades slower than β=4
  - higher optimal threshold (0.61) = more confident peaks
  - β=6 better than β=4 because soft gating needs more aggressive falloff to mimic piecewise threshold

combined α=2,β=4: 0.755 @ epoch15 → 0.926 @ epoch30 → 0.763 @ epoch35
  - sharp degradation after peak, β too low

adapted cornernet: 0.816 @ epoch15 → 0.660 @ epoch26
  - severe patch overfitting, bad

regression cornernet: 0.804 F2 peak
  - log(1-error) instead of log(pred) for pos case
  - underperforms probably because gradient signal too weak
  - (target-pred) focal term already bounds the target, log(1-error) is redundant

best performers: fuzzy α=2,β=4, combined α=2,β=6 (both hit 0.926 peak)
need full degradation sweep on fuzzy to compare stability



● Summary: Random Sampling Pipeline Failure

  Problem

  New random sampling pipeline produces models that fail at inference despite good training metrics. Old strided sampling pipeline works.

| Metric              | Random Sampling (115% data) | Random Sampling (225% data) | Strided Sampling |
  |---------------------|-----------------------------|-----------------------------|------------------|
  | Training peak_dist  | 0.936                       | ~0.94                       | 0.864            |
  | Training peak_sharp | 0.941                       | ~0.94                       | 0.783            |
  | Patch-level invariance | Poor                     | Good                        | Moderate         |
  | Conf @ 0.5 overlap (tomo_00e047) | -            | 0.30                        | -                |
  | Conf @ 0.6 overlap (tomo_00e047) | -            | 0.85                        | -                |
  | Inference F2 @ 0.5 overlap | 0.00               | 0.60                        | 0.822            |
  | Inference F2 @ 0.6 overlap | -                  | 0.74                        | -                |

  Concrete Findings

  1. Translation Invariance
  - Random model (115% data): Only exact multiples of 16 work. Offsets 1-15 → predictions crash to 0.004
  - Random model (225% data): Good patch-level invariance (0.84-0.99 across all offsets), BUT still fails at whole-tomogram inference with standard overlap
  - Strided model: Multiples of 8 work. Bad offsets still give 0.01-0.17 (graceful degradation)
  - Strided Z axis almost fully invariant (0.83-1.0 at ALL offsets)

  2. The Overlap Paradox
  - Strided model: Works at 0.5 overlap (F2=0.822), despite worse patch-level invariance
  - Random model (225%): Needs 0.6+ overlap to achieve decent performance (F2=0.74), despite better patch-level invariance
  - **Core mystery**: Better individual patches compose worse into full tomograms

  3. Generalization
  - Strided 0.95 fraction train got 0.6 F2 on HELD OUT val, no dropout no augmentation - proves generalization
  - Random model shows excellent metrics on training patches but poor composition at inference

    -also gaussian size isnt really an issue, using sigma of 200 vs 300 yielded similarly dogshit results

  Refuted Theories

  | Theory                                 | Why Refuted                                                              |
  |----------------------------------------|--------------------------------------------------------------------------|
  | Data volume                            | Random has 15-125% MORE data than strided (EDIT: so uh 125% data does seem to perform better very obviously, SEE ABOVE)|
  | CPU vs CUDA / AMP                      | Both pipelines use identical inference code                              |
  | Grid alignment / position memorization | Old model got 0.6 F2 on held-out val - genuinely generalized, doesnt seem to be a straight memorization thing             |
  | Hyperparameters                        | Matched LR, batch, accum to old pipeline - still underperforms           |
  | Data extraction bug                    | Verified patches are byte-identical (max diff = 0.0)                     |
  | Translation invariance alone           | 225% random has BETTER patch invariance but WORSE tomo composition       |

  Unexplained

  - Why does strided produce mod-8 tolerance when stride is 40/72/72?
  - Why does random model (225%) have excellent patch-level invariance but require 0.6+ overlap for decent performance?
  - Why does strided with worse patch-level behavior compose better at lower overlap (0.5)?
  - What property of strided sampling teaches composition/voting behavior that random doesn't?

  Compelling (But Incomplete) Theory: Implicit Augmentation from Max-Pool Target Generation

  **Target shape differences:**
  - Old pipeline: gaussian at full-res then max_pool → target shape varies depending on motor's sub-grid position (1.0 peak, 0.9 adjacent due to maxpool edge effects). creates fuzzier targets, to cornernet loss (>= 0.9 threshold to be considered pos sample) it creates 1x1, 2x1, 2x2, etc shaped peaks. so model learns to output (1x1, 2x1, 2x2) peaks with some falloff logits (noteably missing from the new pipeline)
  - New pipeline: gaussian directly in ds-space → always tight 1x1 peaks at exact grid positions

  **What this explains:**
  - Why random needs 0.6+ overlap: Sharp 1x1 peaks may not vote/average well, need more overlap to smooth (though the random one is more translation invariant so?)
  - Why strided works at 0.5 overlap: Fuzzy blobs might aggregate better (kinda hand wavy due to strided having worse translation invariance)
    - dice difference comes from outputting fuzzier blobs

  **What this DOESN'T explain:**
  - Why strided has worse patch-level invariance (mod-8 tolerance) but better tomo-level composition
  - The specific mod-8 pattern in strided when stride is (40,72,72)
  - Why fuzzy blobs compose better despite being theoretically less precise and being less translation invariant
  - How strided generalizes to held-out validation despite seeming less robust

  **Status:** Compelling explanation for overlap requirements and dice differences, but doesn't solve the core composition mystery.

to test: generate blobs similar to the maxpooling approach (if near an edge in DS space then that neighboring voxel gets super high weigting maxpool like behavior)
hopefully though there is an easy way to do this without resorting to maxpooling retarded shit so
use 0.75 cornernet threshold, angstrom sigma 250, 

position (x), dist to pixel 11 center, 250 angstrom
10.75               0.75                    0.74
10.90               0.60                    0.83

so essentially if we want multi pixel gaussians considered by cornernet loss, it needs to be about 0.7 ds_pixels away from the center of another one, or 0.2 ds_pixels away from edge
might still be pretty conservative but this seems to be a pretty major reason why the other one learned so well (the other one used 0.9 threshold but with max pooling so math works out to about 0.1 ds_pixels in that)

IF THIS IS THE TRUE MECHANIC BEHIND WHY, we find that fuzzier gaussians (more than 1 pixel) when stitched for inference outperform strictly 1x1 gaussians even when 1x1 gaussians have more overlap (0.5 vs 0.6, about 2x more inference compute)

ALSO AFTER THIS LETS ADD SOME VISUALIZATION TO WHAT THE CONVERT_PT script is actually doing with the heatmap stuff so its not just a black box