# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 23:36:52 2018

@author: Yilin Liu
"""

import tv2d, mctv2d
import parameter as pa

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float

def psnr(X, Y):
    MAX = 1
    M, N = np.shape(X)
    mse = (np.sum((X - Y) * (X - Y))) / (M * N)
    return 10 * np.log10(MAX * MAX / mse)

def sub_plot_org():
    plt.subplot(3, 5, 2)
    plt.imshow(img, cmap = plt.cm.gray)
    plt.axis('off')
    plt.title('Original Image', fontsize = 'small')

def sub_plot_res(num, X, title):
    plt.subplot(3, 5, num)
    plt.imshow(X, cmap = plt.cm.gray)
    plt.axis('off')
    plt.title(title + ', PSNR = %.4f'%psnr(X, img), fontsize = 'small')

def sub_plot_error_img(num, X, min, max):
    """
    This function is used to plot error images of three denoising methods.
    [min, max] is the relative error range of pixels.
    """
    plt.subplot(3, 5, num)
    plt.imshow(np.fabs(X - img), vmin = min, vmax = max, cmap = plt.cm.gray)
    cb = plt.colorbar(aspect = 8, shrink = 1, pad = 0.05, orientation = 'horizontal')
    cb.ax.tick_params(labelsize = 'small')
    cb.set_ticks(np.arange(min, max + 0.1, 0.1))
    plt.axis('off')

# load demo synthetic block image and demo noisy image
load_file_name = 'image'
image = sio.loadmat(load_file_name)
img = image['img']

load_file_name = 'noi_image'
noi_image = sio.loadmat(load_file_name)
noi_img = noi_image['noi_img']

# set parameter values for TV and MCTV
"""
lamda is the regularization parameter in TV and MCTV denoising models;
rho can be called Lagrangian coefficient which is introduced by ADMM algorithm.
The optimized values of this two hyper-parameters are chosen through experiments.
K, err are both for the convergence condition in TV and MCTV.
K is the maximum number of iterations;
err is the error (measured by Euclidean distance)
between results of two adjacent iterations.
alpha_ratio is for the nonconvexity parameter alpha.
For TV, alpha_ratio is 0 (actually it won't be used in my code);
for MCTV, I usually set it between 0.3 and 0.7.
"""
lamda = 0.05
rho = 100
K = 200
err = 0.0001
para_tv = pa.Parameter(lamda, rho, K, err, 0)

lamda = 0.1
rho = 50
K = 200
err = 0.0001
alpha_ratio = 0.5
para_mctv = pa.Parameter(lamda, rho, K, err, alpha_ratio)

# denoising
X1 = tv2d.denoising_2D_TV(noi_img, para_tv)
X2 = img_as_float(mpimg.imread('NLTV.jpg'))
X3 = mctv2d.denoising_2D_MCTV(noi_img, para_mctv)

# plot demo result figure
plt.figure()
plt.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.95,
                        wspace = 0.05, hspace = 0.05)

sub_plot_org()
sub_plot_res(4, noi_img, 'Noisy Image')
sub_plot_res(6, X1, 'TV')
sub_plot_res(8, X2, 'NLTV')
sub_plot_res(10, X3, 'MCTV')
sub_plot_error_img(11, X1, 0, 0.2)
sub_plot_error_img(13, X2, 0, 0.2)
sub_plot_error_img(15, X3, 0, 0.2)