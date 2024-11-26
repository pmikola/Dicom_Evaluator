import io
import os
import cv2
import numpy as np
import matplotlib
import cv2 as cv
from skimage.feature import match_template, peak_local_max
from skimage.transform import rescale, resize, downscale_local_mean
from matplotlib import pyplot as plt
from pydicom.data import get_testdata_file
import pydicom
from PIL import Image
import time
from PIL import Image, ImageEnhance
import scipy.signal
import zipfile
from brisque import BRISQUE
# import MTM
# from MTM import matchTemplates
matplotlib.use('TkAgg')

base = 'C:\PRACA\Depi\Prg_prj\Dicom_Evaluator\Dicom_Evaluator\CALIBRATION_IMG\\'
folder_1 = 'fotki_do_kalibracjI\\'
folder_2 = 'testy_do_kalibracji_obrazek_testowy\\'

start = time.time()
def show_all_pic(images, disp):
    if disp == 1:
        fig = plt.figure(figsize=(6, 6))
        rows = 5
        columns = 4
        for i in range(len(images)):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(images[i].astype('uint8'))
            plt.axis('off')
        plt.tight_layout(h_pad=1, w_pad=1)
        plt.show()
    else:
        pass

def getnshow_dicom(ds, rgb, disp):
    images = []
    for i in range(ds.pixel_array.shape[0]):
        if rgb == 'r':
            ds.pixel_array[i, :, :, 2] = 0
            ds.pixel_array[i, :, :, 1] = 0
        if rgb == 'b':
            ds.pixel_array[i, :, :, 0] = 0
            ds.pixel_array[i, :, :, 1] = 0
        if rgb == 'g':
            ds.pixel_array[i, :, :, 2] = 0
            ds.pixel_array[i, :, :, 0] = 0
        else:
            pass
        images.append(ds.pixel_array[i, :, :, :])
    show_all_pic(images, disp)
    return images


def image_avg(images):
    avg_image = images[0]
    for i in range(len(images)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(images[i], alpha, avg_image, beta, 0.0)
    return avg_image


# Note: Import DCM
def zips2dcm(base,path):
    path_f = base+path
    items_p1 = os.listdir(path_f)
    fn = [item for item in items_p1 if item.endswith('.zip') and os.path.isfile(os.path.join(path_f, item))]
    if len(fn) > 1:
        for item in fn:
            file_name = os.path.abspath(path_f+item)
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(path_f)
            zip_ref.close() #
            os.remove(file_name)
        return path_f, [item[0:-4]+'.dcm' for item in  fn]
    else: return path_f, [item for item in items_p1 if item.endswith('.dcm') and os.path.isfile(os.path.join(path_f, item))]
ph1,f1 = zips2dcm(base,folder_1)
ph2,f2 = zips2dcm(base,folder_2)
# Note: Import DCM
disp = 1

def read_dcm(ph,f,channel='col',disp=0):
    filename = pydicom.data.data_manager.get_files(ph, f)[0]
    dsx = pydicom.dcmread(filename, force=True)
    images_col = getnshow_dicom(dsx, channel, disp)
    return images_col

# Note: Brisque method
# Attention: This score should be from 0 to 100 where 0 is best perceived image
brisque = BRISQUE(url=False)
imgs = read_dcm(ph2, f2[25],channel='col',disp=1)
for img in imgs:
    print(brisque.score(img= img))


end = time.time()
print("Evaluation time: ", end - start)

