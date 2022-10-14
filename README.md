# crop_and_deskew

##Prerekvizity:

numpy, multiprocessing, pyFFTW, skimage, cv2

> pip install pyFFTW 
> pip install scikit-image
> pip install opencv-contrib-python

Použití viz metoda:
 > crop_and_deskew(img)
 
Nezbytné metody: 
 * deskew_fft
 * get_pixels_on_line
 * rotate
 * crop_image

