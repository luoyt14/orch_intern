import os
import tqdm
import shutil


NEW_PATH = "Picture_voc/SegmentationClassPNG"
OLD_PATH = "Picture_segformer_1"
ABNOEMAL_LIST = "Picture_abnormal.txt"

with open(ABNOEMAL_LIST) as f:
    for item in tqdm.tqdm(f.readlines()):
        filename = item.strip()
        shutil.copy(os.path.join(NEW_PATH, filename[:-3]+"png"), os.path.join(OLD_PATH, filename))
