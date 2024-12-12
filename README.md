
## Data Format

### Hameln
- seg4cls_train
  - sample   
    - [PatchID]_[Year]_[GridX]_[GridY]_[Category].png  (Input RGB image)
  - gt
    - [PatchID]_[GridX]_[GridY].png (Ground-Truth labels, bg=0, agr=1, for=2, wat=3, set=4)

### LGLN
- train
  - maps
    - __[PatchID]_[Year].tif__ (Input RGB maps from year 1898 to 2017, including patch 3821, 3822, 3921)
    - __[PatchID]\_[Year]\_[Class].tif__ (Ground-Truth labels from year 1973 to 1975, including patch 3821, 3822, 3921 and class FlowingWater, Grassland, Settlement, StandingWater ad Woodland. 
    In the label map, 0 represents background, 1 the foreground class and 3 represents the border of the map. 
    Note that there is one map reads with 0 and an outflow number (2**8), it is ignored as there is no foreground pixels in this map)
- test
  - __[PatchID]_[Year].tif__ (Input RGB maps from year 1898 to 2017 of patch 3922)
  - __[PatchID]\_[Year]\_[Class].tif__ (label maps from year 1898 to 1996 of patch 3922)

## Experiment
### E1(All): train with maps and label from patch 3821, 3822, 3921 and all ages
- train config file : 
    ```yaml
    DATASET:
      # ...
      start_year: 2023 # or 1973
      end_year: 2023 # or 1975
      train_mode: all
    TRAIN:
      # ...
      max_epoch: 20
      optimizer:
        lr: 0.0002
        # ...
      lr_scheduler:
        milestones: [10, 15]
        # ...
    ```
  
### E2(Pre): pre-train with maps and label from patch 3821, 3822, 3921 and the anchor years (1973-1975 or 2023)
- train config file : 
    ```yaml
    DATASET:
      # ...
      start_year: 2023 # or 1973
      end_year: 2023 # or 1975
      train_mode: pre
    TRAIN:
      # ...
      max_epoch: 20
      optimizer:
        lr: 0.0002
        # ...
      lr_scheduler:
        milestones: [10, 15]
        # ...
    ```

### E2(Trace): Fine-tuning the pre-trained model.
- train config file : 
    ```yaml
    DATASET:
      # ...
      start_year: 2023 # or 1973
      end_year: 2023 # or 1975
      train_mode: trace
    TRAIN:
      # ...
      max_epoch: 3
      optimizer:
        lr: 0.00002
        # ...
      lr_scheduler:
        milestones: [3]
        # ...
    ```
  
- test
  - patch 3922 in year 1898, 1974, 1982, 1996.