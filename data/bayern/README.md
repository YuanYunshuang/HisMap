# Documentation for Bayern Maps

## File Location and Info
| Where                                                                      | What                                                                                    | More                                                                                                                                                                                                            |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MAP] IKG:O:\forschung\BKG-Gauss\dat\Bayern\TK_Donauwörth\tk_entzerrt      | Map sheets                                                                              |                                                                                                                                                                                                                 |
| [LBL] IKG:O:\forschung\BKG-Gauss\dat\Bayern\TK_Donauwörth\tk_reference\tif | Labels                                                                                  | Name: [Mapsheet]\_[Year]\_[Class].tif, 1: class true, 0: class fals, 3: outsite mapsheet. Class==testsite: labeled area. Year=1910: 1910--1939. Year==1959: 1959                                                |
| [MAP_LBL_ori] ISI:/home/yuan/data/HisMap/Bayern/original                   | Map sheets and labels manually selected from  [MAP] and [LBL]                           | maps: selected map sheets. lbls: selected labels                                                                                                                                                                |
| [MAP_LBL_prc] ISI:/home/yuan/data/HisMap/Bayern/processed                  | Map sheets and labels processed (by `data/bayern/generate_smolsam.py`) from [MAP_LBL_ori] | Cropped the map sheets to testsite area. Merged label files to one with the label mapping: [0: 'Background', 1:'Wald', 2:'Wiese', 3:'Siedlung', 4:'Fluss', 5:'Stehendes_gewaesser'] (518_1910 labels are empty) |



## Available Map Sheets
- Map sheets: ['01_489_N_1910', '01_489_N_1939', '01_490_N_1910', '01_490_N_1938', '01_518_F_1941', '01_519_N_1910', '02_7230_N_1959_E0', '02_7330_N_1959']
- Labels: ['489_1910', '489_1933-39', '490_1910', '490_1933-39', '518_1933-39', '519_1910', '7230_1959', '7330_1959']
