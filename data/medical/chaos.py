import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image

ds = pydicom.dcmread("/home/yuan/Downloads/CHAOS_Train_Sets/Train_Sets/CT/1/DICOM_anon/i0000,0000b.dcm")          # read metadata + pixel data
arr = ds.pixel_array                        # numpy array (handles many codecs if pylibjpeg installed)

# Apply VOI LUT (windowing), if present
arr = apply_voi_lut(arr, ds)

# Convert to float and rescale to 0–1 for display
arr = arr.astype(np.float32)
arr -= arr.min()
if arr.max() > 0:
    arr /= arr.max()

pil_img = Image.fromarray((arr * 255).astype(np.uint8))  # convert to PIL Image
pil_img.save("/home/yuan/Downloads/tmp.png")               # save image


# Access metadata
print(ds.PatientID, ds.Modality, ds.StudyDate)