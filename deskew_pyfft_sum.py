#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage.filters import threshold_otsu
import skimage.transform
import os
import time
import glob
import pyfftw
import multiprocessing


def get_pixels_on_line(image, alpha, h, c, num_of_points, point_number_fraction=1.0 ):
    ys = np.linspace(c[0] + np.cos(alpha*0.01745)*(-h/8), c[0] + np.cos(alpha*0.01745)*(h/8), num_of_points, dtype=np.int32)
    xs = np.linspace(c[1] - np.sin(alpha*0.01745)*(-h/8), c[1] - np.sin(alpha*0.01745)*(h/8), num_of_points, dtype=np.int32)   
    return ys, xs

def deskew_fft(input_image, angle_range=10, step=0.25, point_number_fraction=1.0, downsample_constant=1.0, crop_percent=0.0):
    image = input_image.copy()
 
    if (downsample_constant > 1.0):
        image = cv2.resize(image, (int(image.shape[1]//downsample_constant), int(img.shape[0]//downsample_constant)))
    if (crop_percent > 0.0):
        x = int(image.shape[1] * (crop_percent/100.0))
        y = int(image.shape[0] * (crop_percent/100.0))
        image = crop_image(image)[y:-y, x:-x]
        #$print(image.shape)
    h = image.shape[0]
    w = image.shape[1]
    num_of_points = int(np.floor(np.sqrt((h/4)**2+(w/4)**2)*point_number_fraction))       
    mult_coef = int(1/step)
    start_fft = time.time()
    aa = pyfftw.byte_align(image, dtype='complex64')
    af= pyfftw.empty_aligned(image.shape, dtype='complex64')
    #fft_object_c = pyfftw.FFTW(aa, af, axes=(0,1), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD',threads=multiprocessing.cpu_count())
    fft_object_c = pyfftw.FFTW(aa, af, axes=(0,1), flags=('FFTW_ESTIMATE',), direction='FFTW_FORWARD',threads=multiprocessing.cpu_count())    
    ft = fft_object_c(aa)
    ftshift = np.fft.fftshift(ft)
    stop_fft = time.time()        
    c = [h//2, w//2]     
    print(c[0] - h//7, c[0] + h//7, c[1] - w//7, c[1] + w//7)
    spek = 20 * np.log(np.abs(ftshift[c[0] - h//7: c[0] + h//7, c[1] - w//7: c[1] + w//7]))
    c2 = [spek.shape[0]/2, spek.shape[1]/2]


    angle = 0
    angles = []
    start_angle = time.time()
    pixel_array = np.zeros((2*angle_range*mult_coef, num_of_points))
   
    for i in range(-angle_range*mult_coef,angle_range*mult_coef):
        alpha = i/mult_coef
        angles.append(alpha)
        ys, xs = get_pixels_on_line(spek, alpha, h, c2, num_of_points, point_number_fraction)
        pixel_array[i + angle_range*mult_coef, :] = spek[ys,xs]
    stop_angle = time.time()
    idx = np.argmax(np.sum(pixel_array, axis=1))
    print(f"FFT time: {stop_fft-start_fft}, ANGLE time {stop_angle - start_angle}, IDX {idx}")    
    angle = -angle_range + step*idx      
    if angle == -angle_range:
        angle = 0.0      
    return angle


def rgb_binary(image):
    if len(image.shape) != 3:
        return False
    thresh_r = threshold_otsu(image[:, :, 0])
    thresh_g = threshold_otsu(image[:, :, 1])
    thresh_b = threshold_otsu(image[:, :, 2])
    binary_r = image[:, :, 0] > thresh_r
    binary_g = image[:, :, 1] > thresh_g
    binary_b = image[:, :, 2] > thresh_b
    binary = np.dstack([binary_r, binary_g, binary_b])
    result = np.uint8(np.sum(binary, axis=2) > 0) * 255
    return result

def rotate(image, angle):
    return skimage.transform.rotate(image, angle)

def rotate_slow(image, angle):
    temp_image = np.ones([image.shape[0] * 2, image.shape[1] * 2], dtype=np.uint8) * np.mean(image)
    ymin = int(temp_image.shape[0] / 2.0 - image.shape[0] / 2.0)
    ymax = int(temp_image.shape[0] / 2.0 + image.shape[0] / 2.0)
    xmin = int(temp_image.shape[1] / 2.0 - image.shape[1] / 2.0)
    xmax = int(temp_image.shape[1] / 2.0 + image.shape[1] / 2.0)
    temp_image[ymin: ymax, xmin: xmax] = image
    temp_image = skimage.transform.rotate(temp_image, angle)
    image = temp_image[ymin: ymax, xmin: xmax]
    return image    

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

if __name__ == "__main__":
    files = sorted(glob.glob("/home/neduchal/Dokumenty/Data/Naki/deskew_set/*.jpg"))
    print(files)
    for i, f in enumerate(files):
        src = cv2.imread(f, 0)
        blur = cv2.blur(src, (5, 5))
        img = src.copy()
        start = time.time()
        angle = deskew_fft(blur, step=0.25)
        print(os.path.basename(f), angle, time.time() - start)
        cv2.imwrite("../data/output/pyfftw/"+os.path.basename(f), crop_image(rotate(src, angle), 50))
        if i > 4:
            break
