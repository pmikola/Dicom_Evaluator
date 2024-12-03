import io
import os
import cv2
import numpy as np
import matplotlib
import cv2 as cv
import torch
import torcheval
import torchvision
from ignite.engine import Engine
from ignite.metrics import PSNR
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
import torch.hub as h
from torchvision.transforms import transforms
from torcheval.metrics import PeakSignalNoiseRatio

# import MTM
# from MTM import matchTemplates
matplotlib.use('TkAgg')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
base = 'C:\PRACA\Depi\Prg_prj\Dicom_Evaluator\Dicom_Evaluator\CALIBRATION_IMG\\'
folder_1 = 'fotki_do_kalibracjI\\'
folder_2 = 'testy_do_kalibracji_obrazek_testowy\\'
folder_3 = 'fotki_do_kalibracji_NANOCHECKER\\'

class AddNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

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



def read_dcm(ph,f,channel='col',disp=0):
    filename = pydicom.data.data_manager.get_files(ph, f)[0]
    dsx = pydicom.dcmread(filename, force=True)
    images_col = getnshow_dicom(dsx, channel, disp)
    return images_col

# Note: Brisque method
# Attention: This score should be from 0 to 100 where 0 is best perceived image

# Note: Brisque method
brisque = BRISQUE(url=False)
def Brisque_score(img,impre1,impre2):
    img_score = torch.tensor(brisque.score(img=img)).unsqueeze(0)
    impre1 = torchvision.transforms.functional.to_pil_image(noise_addition(impre1))
    impre2 = torchvision.transforms.functional.to_pil_image(impre2)
    imgp1_score = torch.tensor(brisque.score(img=impre1)).unsqueeze(0)
    imgp2_score = torch.tensor(brisque.score(img=impre2)).unsqueeze(0)
    b_scores = 1 / torch.std(torch.cat([img_score,imgp1_score, imgp2_score], dim=0))
    return b_scores

# Note: ARNIQA method
# print(torch.__version__)
model = h.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                       regressor_dataset="kadid10k").to(device)
model.eval()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #AddNoise(0.,0.1),

])

noise_addition = transforms.Compose([
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    AddNoise(0.,0.2),

])
# print(model)
# Note: ARNIQA method

# Note: Frequency distribution metric
def freq_diversity_metric(img,impre1,impre2,n_regions=20):
    im = torch.from_numpy(img).float().to(device)
    fft_im = torch.fft.fftn(im)
    fft_imp1 = torch.fft.fftn(noise_addition(impre1).permute(1,2,0))
    fft_imp2 = torch.fft.fftn(impre2.permute(1,2,0))
    fft_shift_im = torch.fft.fftshift(fft_im)
    fft_shift_imp1 = torch.fft.fftshift(fft_imp1)
    fft_shift_imp2 = torch.fft.fftshift(fft_imp2)
    spectrum = torch.abs(fft_shift_im)
    spectrum_p1 = torch.abs(fft_shift_imp1)
    spectrum_p2 = torch.abs(fft_shift_imp2)
    height, width = spectrum.shape[-3], spectrum.shape[-2]
    region_height = height // n_regions
    region_width = width // n_regions
    region_energies = []
    region_energies_p1 = []
    region_energies_p2 = []

    for i in range(n_regions):
        for j in range(n_regions):
            start_h = i * region_height
            end_h = (i + 1) * region_height
            start_w = j * region_width
            end_w = (j + 1) * region_width
            region_energy = torch.mean(spectrum[start_h:end_h, start_w:end_w,:])
            region_energy_p1 = torch.mean(spectrum_p1[start_h:end_h, start_w:end_w,:])
            region_energy_p2 = torch.mean(spectrum_p2[start_h:end_h, start_w:end_w,:])
            region_energies.append(region_energy)
            region_energies_p1.append(region_energy_p1)
            region_energies_p2.append(region_energy_p2)


    for i in range(len(region_energies)):
        for j in range(i + 1, len(region_energies)):
            diversity_score = 1/( torch.mean(
                torch.var(torch.cat([
                torch.abs(region_energies[i] + region_energies[j]).unsqueeze(0),
                torch.abs(region_energies_p1[i] + region_energies_p1[j]).unsqueeze(0),
                torch.abs(region_energies_p2[i] + region_energies_p2[j]).unsqueeze(0)], dim=0)))+1e-6)

    return diversity_score

# Note: Frequency distribution metric

# Note: Gradient similarity metric
def grad_difference_metric(img, impre1, impre2, n_regions=20):
    im = torch.from_numpy(img).float().to(device)
    imp1=noise_addition(impre1).permute(1, 2, 0)
    imp2=impre2.permute(1, 2, 0)

    _,grad_im = torch.gradient(im,dim=[0,1])
    _,grad_imp1 = torch.gradient(imp1,dim=[0,1])
    _,grad_imp2 = torch.gradient(imp2,dim=[0,1])
    grad_mse = torch.nn.functional.mse_loss(grad_im,grad_imp1) + torch.nn.functional.mse_loss(grad_im,grad_imp2)
    grad_div_score = grad_mse.mean()
    return grad_div_score

# Note: Gradient similarity metric

# Note: PSNR metric
psnr_metric = PeakSignalNoiseRatio(data_range=255.)

def psnr_score(img,impre1,impre2):
    img_score = torch.tensor(brisque.score(img=img)).unsqueeze(0)
    impre1 = torchvision.transforms.functional.to_pil_image(noise_addition(impre1))
    impre2 = torchvision.transforms.functional.to_pil_image(impre2)
    psnr_metric.update(img_score,impre1)
    psnr_score_p1 = psnr_metric.compute()
    psnr_metric.update(img_score, impre2)
    psnr_score_p2 = psnr_metric.compute()
    imgp1_score = psnr_score_p1.unsqueeze(0)
    imgp2_score = psnr_score_p2.unsqueeze(0)
    psnr_score = 1 / torch.var(torch.cat([imgp1_score, imgp2_score], dim=0))
    return psnr_score
# Note: PSNR metric


ph,f = zips2dcm(base,folder_2)
# Note: Import DCM
disp = 1
psnr_scores = []
BRISQUE_scores = []
ARNIQA_scores = []
fft_diversity_scores = []
grad_diff_scores = []
imgs = read_dcm(ph, f[12],channel='col',disp=1)
i = 0
with torch.no_grad():
    for img in imgs:
        print("Image Number: ", i+1,"\n>------------------------------>")
        im1 = preprocess(img).float().to(device)
        im2 = preprocess(img).float().to(device)

        psnr_score = psnr_score(img, im1, im2)
        psnr_scores.append(psnr_score)
        print("PSNR Score          : ", psnr_score)

        BRISQUE_score = Brisque_score(img, im1, im2)
        BRISQUE_scores.append(BRISQUE_score)
        print("BRISQUE Score       : ", BRISQUE_score)

        freq_div_Score = freq_diversity_metric(img, im1, im2)
        fft_diversity_scores.append(freq_div_Score)
        print("Freq Diversity Score: ", freq_div_Score)

        grad_div_Score = grad_difference_metric(img, im1, im2)
        grad_diff_scores.append(grad_div_Score)
        print("Grad Diversity Score: ", grad_div_Score)

        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
        ARNIQA_score = model(im1,im2, return_embedding=False, scale_score=True)
        ARNIQA_scores.append(ARNIQA_score)
        print("ARNIQA Score        : ",ARNIQA_score, "\n<------------------------------<\n")
        i+=1


end = time.time()
print("Evaluation time: ", round(end - start,2) , " [s]")
print("Best Image Number BRISQUE:",torch.argmax(torch.FloatTensor(BRISQUE_scores)).item())
print("Best Image Number ARNIQA:",torch.argmax(torch.FloatTensor(ARNIQA_scores)).item())
print("Best Image Number FREQ DIV:",torch.argmax(torch.FloatTensor(fft_diversity_scores)).item())
print("Best Image Number GRAD DIFF:",torch.argmax(torch.FloatTensor(grad_diff_scores)).item())
print("Best Image Number PSNR:",torch.argmax(torch.FloatTensor(psnr_scores)).item())

