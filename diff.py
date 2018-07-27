# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:09:22 2018

@author: Yilin Liu
"""

import numpy as np

def Dx(U):
    
    M, N = np.shape(U)
    D = np.zeros((M, N))
    D[:, 1: N] = U[:, 1: N] - U[:, 0: N - 1]
    D[:, 0] = U[:, 0] - U[:, N - 1]
    
    return D

def Dxt(U):
    
    M, N = np.shape(U)
    D = np.zeros((M, N))
    D[:, 0: N - 1] = U[:, 0: N - 1] - U[:, 1: N]
    D[:, N - 1] = U[:, N - 1] - U[:, 0]
    
    return D

def Dy(U):
    
    M, N = np.shape(U)
    D = np.zeros((M, N))
    D[1: M, :] = U[1: M, :] - U[0: M - 1, :]
    D[0, :] = U[0, :] - U[M - 1, :]
    
    return D

def Dyt(U):
    
    M, N = np.shape(U)
    D = np.zeros((M, N))
    D[0: M - 1, :] = U[0: M - 1, :] - U[1: M, :]
    D[M - 1, :] = U[M - 1, :] - U[0, :]
    
    return D