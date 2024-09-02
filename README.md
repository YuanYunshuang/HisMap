
## Data Format

- Hameln
  - seg4cls_train
    - sample   
      - [PatchID]_[Year]_[GridX]_[GridY]_[Category].png  (Input RGB image)
    - gt
      - [PatchID]_[GridX]_[GridY].png (Ground-Truth labels, bg=0, agr=1, for=2, wat=3, set=4)