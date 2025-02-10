import os

import numpy as np
from src.utils.misc import load_json
import glob

files = glob.glob("/home/yuan/code/HisMap/data/llm/*.json")
tp = np.array([0, ] * 4)
invalide = 0
for f in files:
    data = load_json(f)
    basename = os.path.basename(f).replace(".json", ".jpg")
    if basename not in data:
        invalide += 1
        continue
    gt = basename.split(".")[0].split("_")[-1]
    v = data[basename]
    for i in range(4):
        try:
            if int(gt[i]) == v['cls'][i]:
                tp[i] += 1
        except:
            invalide += 1
            print(basename)

print(invalide)
print(tp / (len(files) - invalide))