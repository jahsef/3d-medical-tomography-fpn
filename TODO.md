


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
