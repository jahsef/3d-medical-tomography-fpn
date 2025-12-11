


generate auto test running
structure can be like
TEST_NAME
-run1
--weights
--logs
-run2
....
-run3
...

  1. Refactor CONFIG → config.json loading in train.py
  2. Create orchestrator script that:
    - Generates config variants (different loss functions, etc.)
    - Creates directory structure
    - Launches train.py processes
  3. Post-processing script to collate CSVs into comparison tables


mse
vanilla bce
(wing loss or other paper stuff)
continuous focal
cornernet
cornernet beta scheduling


MOVE GAUSSIAN GENERATION AWAY FROM DATASET
precompute them like patches as well
may want to move angstrom sigma to 240 after moving away from maxpool interp to 1/16 (extends sigma to angstrom_sigma/16 + 8 ish)
old approach
generate gaussian in realpixel, maxpool to 1/16 (adds about 8 to the sigma in realpixel)

old grid stride patch creation was poo and inefficient

new positive mining strat (no centering distance constraint, monai inferer can handle edge artifacts for me)
with patch (d,h,w)
given motor position (z,y,x) we generate boundaries (z,y,x) +- (d,h,w) for starting locations of the patch
(could use like 0.95(d,h,w) but again, letting monai handle it should be fine)

NEW PROPOSED HARD NEGATIVE MINING IN PATCH CREATION SCRIPT
Motor at position M = (z, y, x)

Exclusion zone: M ± (1.1 × patch_size)  // motor + margin
  → Ensures motor isn't in patch or on boundary

Proposal zone: M ± (2.0 × patch_size)  // where to sample from
  → Gives range around motor to sample hard negatives

Sample random patch top-left in: [proposal_zone - exclusion_zone]
  → Random location near motor but guaranteed not containing it

Prune: patches that go outside tomogram bounds

easy (random) negatives
just randomly sample negatives not too hard lol
still need to use the hard negative exclusion zone constraint but otherwise much more free to sample