
#Import the required modules

import numpy as np
import cv2, os

#load imagesn
#segment and normalize
#Compute features: - Zero crossing Wavelet - Local Binary Patterns (LBP) - Gabor 2D Wavelets - Laplacian of Gaussian
#Compute metrics for Iris verification (hamming distance)
#Fusion of score / Multiple signatures
#Iris identification

class Cassia:

    def load_images(self):
        