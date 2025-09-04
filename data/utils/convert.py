import os
import json
import cv2

from imgio import read_tiff


def tiff_to_png(tif_file, scale=0.25, out_dir=None):
    step = int(1 / scale)
    TIF = read_tiff(tif_file)
    TIF = TIF[::step,::step]
    if out_dir is None:
        cv2.imwrite(tif_file.replace('.tif', '.png'), TIF[..., ::-1])
    else:
        cv2.imwrite(out_dir + '/' + os.path.basename(tif_file).replace('.tif', '.png'), TIF[..., ::-1])



def json_to_geojson(json_file, overlap_date):
    patch, year = os.path.basename(json_file).split('.')[0].split('-')
    tfw_file = f"/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m/{patch}_{year}.tfw"
    with open(tfw_file, 'r') as f:
        lines = f.readlines()
        tfw_x = float(lines[4].strip())
        tfw_y = float(lines[5].strip())

    # convert the json files of haowei's labels into geojson
    geojson_tpl = {
      "type": "FeatureCollection",
      # "name": f"{patch}_{year}",
      "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25832"}},
      "features": []
    }
    def feature_tpl(plg_type, plg_id, polygon):
        cls = 1 if plg_type == 'woodland' else 2
        return {
        "type": "Feature",
        "properties": {"type": plg_type, "id": plg_id, "cls": cls},
        "geometry": {
          "type": "MultiPolygon",
          "coordinates":[[polygon]]
        },
      }

    with open(json_file, 'r') as f:
        data = json.load(f)
        for d in data['objects']:
            if d['date'] not in overlap_date:
                overlap_date.append(d['date'])
                plg_type = 'woodland' if d['label'] == 'static' else d['label']
                polygon = []
                for p in d['polygon']:
                    polygon.append([ tfw_x + p[0],  tfw_y - p[1]])
                geojson_tpl['features'].append(feature_tpl(plg_type, d['id'], polygon))
    with open(json_file.replace('json', 'geojson'), 'w') as f:
        json.dump(geojson_tpl, f, indent=4)


def read_jsons(path):
    overlap_date = []
    for f in sorted(os.listdir(path)):
        json_to_geojson(os.path.join(path, f), overlap_date)

# with open("/home/yuan/tttt.geojson", 'r') as f:
#     data = json.load(f)
#     print(len(data['features']))
# path = '/home/yuan/data/HisMap/Haowei/json'
# overlap_date = []
# filelist = [
#     # "3922-1898", "3922-1925", "3922-1936", "3922-1945", "3922-1952", "3922-1960",
#     "3821-1974", "3821-1898", "3821-1919"
# ]
# for f in filelist:
#     json_to_geojson(os.path.join(path, f + '.json'), overlap_date)

tiff_to_png(
    "/home/yuan/data/HisMap/LGLN/dataset/test/maps/3922_1996.tif",
    0.1, "/home/yuan/Downloads"
)
