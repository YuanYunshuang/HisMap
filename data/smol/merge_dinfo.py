import json
import os
import sys


def merge_dinfo(out_dir, *files):
    dinfo = {}
    for f in files:
        if os.path.exists(f):
            with open(f) as json_file:
                data = json.load(json_file)
                dinfo.update(data)
    with open(os.path.join(out_dir, 'dinfo_all.json'), 'w') as json_file:
        json.dump(dinfo, json_file, indent=4)


def merge_sample_list(out_dir, *files):
    samples = []
    for f in files:
        if os.path.exists(f):
            with open(f) as fh:
                data = fh.readlines()
                samples.extend([x.strip() for x in data])
    with open(os.path.join(out_dir, 'train_all.txt'), 'w') as fh:
        fh.write('\n'.join(samples))


if __name__=='__main__':
    merge_dinfo(
        "/home/yuan/data/HisMap/SMOLdata",
        "/home/yuan/data/HisMap/SMOLdata/dinfo_gauss.json",
        "/home/yuan/data/HisMap/SMOLdata/dinfo_siegfried_train.json",
        "/home/yuan/data/HisMap/SMOLdata/dinfo_siegfried_val.json",
        "/home/yuan/data/HisMap/SMOLdata/dinfo_siegfried_test.json",
    )
    # merge_sample_list(
    #     "/home/yuan/data/HisMap/SMOLdata",
    #     "/home/yuan/data/HisMap/SMOLdata/train.txt",
    #     "/home/yuan/data/HisMap/SMOLdata/train_siegfried.txt"
    # )