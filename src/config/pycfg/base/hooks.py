
train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10),
        dict(type="SegPlotWriterHook", max_n_img=2, log_every=20)
    ]

test_hooks = [
    dict(type='HolisticSegResultHook'),
]



