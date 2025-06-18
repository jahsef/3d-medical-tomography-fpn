from pathlib import Path


path_str = "WindowsPath('C:/Users/kevin/Documents/GitHub/kaggle-byu-bacteria-motor-comp/patch_pt_data/tomo_003acc')"
actual_path = path_str.split("'")[1]

path_obj = Path(actual_path)
print(path_obj)