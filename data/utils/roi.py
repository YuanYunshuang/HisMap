import os
import json

import cv2
import numpy as np
from imgio import read_tiff
from geotiff import GeoTiff


TIF = read_tiff('/home/yuan/data/HisMap/Haowei/newmaps/1960.tif')
cv2.imwrite("/home/yuan/data/HisMap/Haowei/newmaps/1960.png", TIF)

patch_coordinates = {
    "3821": [511316.5, 5783180.5],
    "3822": [522706.5, 5783245.5],
    "3921": [511342.5, 5772058.5],
    "3922": [522756.5, 5772124.5],
}

ROI_coordinates = [521687.0,5768132.0, 531776.0,5775812.0]
print(ROI_coordinates[2] - ROI_coordinates[0])
print(ROI_coordinates[3] - ROI_coordinates[1])

P3821x_min = int(ROI_coordinates[0] - patch_coordinates["3821"][0])
P3821y_min = int(patch_coordinates["3821"][1] - ROI_coordinates[3])

P3822x_max = int(ROI_coordinates[2] - patch_coordinates["3822"][0])
P3822y_min = int(patch_coordinates["3822"][1] - ROI_coordinates[3])

P3921x_min = int(ROI_coordinates[0] - patch_coordinates["3921"][0])
P3921y_max = int(patch_coordinates["3921"][1] - ROI_coordinates[1])

P3922x_max = int(ROI_coordinates[2] - patch_coordinates["3922"][0])
P3922y_max = int(patch_coordinates["3922"][1] - ROI_coordinates[1])

#
# def read_tif(filename):
#     geo_tiff = GeoTiff(filename)
#     tiff_array = np.array(geo_tiff.read())
#     return tiff_array
#
# m = read_tif("/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/3922_1960.tif")
# map for year 1960

# maps_1960 = ["3821_1961", "3822_1960", "3921_1961", "3922_1960"]

map_3821_1961 = read_tiff("/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/3821_1961.tif")
map_3821_1961 = map_3821_1961[P3821y_min:, P3821x_min:]
map_3822_1960 = read_tiff("/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/3822_1960.tif")
map_3822_1960 = map_3822_1960[P3822y_min:, :P3822x_max]
map_3921_1961 = read_tiff("/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/3921_1961.tif")
map_3921_1961 = map_3921_1961[:P3921y_max, P3921x_min:]
map_3922_1960 = read_tiff("/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/3922_1960.tif")
map_3922_1960 = map_3922_1960[:P3922y_max, :P3922x_max]

map_1960 = np.ones((int(ROI_coordinates[3] - ROI_coordinates[1]), int(ROI_coordinates[2] - ROI_coordinates[0]), 3))
map_1960[:map_3821_1961.shape[0], :map_3821_1961.shape[1]] = map_3821_1961
map_1960[:map_3822_1960.shape[0], map_1960.shape[1] - map_3822_1960.shape[1]:] = map_3822_1960
map_1960[map_1960.shape[0] - map_3921_1961.shape[0]:, :map_3921_1961.shape[1]] = map_3921_1961
map_1960[map_1960.shape[0] - map_3922_1960.shape[0]:, map_1960.shape[1] - map_3922_1960.shape[1]:] = map_3922_1960

cv2.imwrite("/home/yuan/Downloads/1960.png", map_1960)