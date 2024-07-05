

def get_hameln_cfg(mode='train'):
    return dict(
        dataset='hameln',
        data_path="/home/data/HisMAP/full_map/3922_2017_forest_bdry.tif",
        mode=mode,
        batch_size=64,
        n_workers=4,
        gpus=0,
        img_size=128,
    )

hameln_train = get_hameln_cfg()