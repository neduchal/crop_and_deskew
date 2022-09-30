#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from deskew_pyfft import deskew_fft, rotate
from deskew_pyfft_sum import deskew_fft, rotate
import numpy as np
import cv2
import glob
import os

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img[:,:,2]>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_and_deskew(img):
    angle = deskew_fft(cv2.blur(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), (5,5)), angle_range=5, step=0.25, downsample_constant=0.5, crop_percent=0.0)
    print(angle)
    img2 = rotate(img.copy(), angle)*255
    print(np.min(img2.astype(dtype=np.uint8)), np.max(img2.astype(dtype=np.uint8)))
    fimg = crop_image(img2.astype(dtype=np.uint8).copy(), tol=75)
    return  fimg.astype(dtype=np.uint8)

if __name__ == "__main__":
    input_dir = "./data/0007 oznaƒeno"
    output_dir = "./output/0007"

    files = glob.glob(os.path.join(input_dir, "*.jpg"))

    done = -1

    for i, f in enumerate(files):
        if done < (i*100)//len(files):
            done = (i*100)//len(files)
            print(done)
        img = cv2.imread(f)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(f)), crop_and_deskew(img))
