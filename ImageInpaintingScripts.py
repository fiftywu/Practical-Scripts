import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
import cv2
import skimage.measure 


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, dmask=0, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    
    # if add a dmask, how to calculate the dmaks region
    dmask = cv2.resize(dmask, dsize=mu1.shape, interpolation=cv2.INTER_NEAREST)
    hole_ssim = np.sum(ssim_map*dmask) / np.sum(dmask)
    nonhole_ssim = np.sum(ssim_map*(1-dmask)) / np.sum(1-dmask)
    
    return np.mean(ssim_map), hole_ssim, nonhole_ssim



class ImageInpaintingMetric():
    def __init__(self, im1, im2, hole=1):
        # im1, im2, 0~1, float, gray, channel=1
        # hole is Matrix, the inpainted region value is 1, channel=1
        self.im1 = im1
        self.im2 = im2
        self.hole = hole
        
    def evaluate(self):
        # overall / hole / unhole
        self.l1 = self.get_l1()
        self.l2 = self.get_l2()
        self.ssim = self.get_ssim()
        self.psnr = self.get_psnr()
    
    def get_l1(self):
        overall = np.mean(np.abs(self.im1-self.im2))
        hole = np.sum(np.abs( (self.im1-self.im2)*self.hole ))/np.sum(self.hole)
        nonhole = np.sum(np.abs( (self.im1-self.im2)*(1-self.hole) ))/np.sum(1-self.hole)
        return np.array([overall, hole, nonhole])
    
    def get_l2(self):
        overall = np.mean((self.im1-self.im2)**2)
        hole = np.sum(self.hole*(self.im1-self.im2)**2)/np.sum(self.hole)
        nonhole = np.sum((1-self.hole)*(self.im1-self.im2)**2)/np.sum(1-self.hole)
        return np.array([overall, hole, nonhole])
        
    def get_psnr(self, L=1):
        [mse, hole_mse, nonhole_mse]=self.get_l2()
        overall = 10 * np.log10(L * L / mse)
        hole = 10 * np.log10(L * L / hole_mse)
        nonhole =  10 * np.log10(L * L / nonhole_mse)
        return np.array([overall, hole, nonhole])
        
    def get_ssim(self, L=1):
        overall, hole, nonhole = compute_ssim(self.im1, self.im2, self.hole, L=L)
        return np.array([overall, hole, nonhole])
    
    def printMeric(self):
        self.evaluate()
        print('L1 [overall/ hole/ non-hole] : ', self.l1,'\n', 
              'L2 [overall/ hole/ non-hole] : ', self.l2,'\n', 
              'SSIM [overall/ hole/ non-hole] : ', self.ssim,'\n', 
              'PSNR [overall/ hole/ non-hole] : ', self.psnr)

if __name__ == "__main__":
    im1 = real_B
    im2 = fake_B
    dmask = dmask
    Metric = ImageInpaintingMetric(im1, im2, dmask)
#     Metric.evaluate()
    Metric.printMeric()
