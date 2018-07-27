# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:59:03 2018

@author: Yilin Liu

Paper: Du, H. & Liu, Y.: Minmax-concave Total Variation Denoising. 
       Signal, Image and Video Processing (2018).
       doi: 10.1007/s11760-018-1248-2
       
Algorithm for arg_min_X 0.5|Y - X|_2^2 + lamda*|X|_MCTV (ADMM)
"""

import numpy as np
import diff

def denoising_2D_MCTV(Y, para):
    
    M, N = np.shape(Y)
    X0 = np.zeros((M + 2, N + 2))
    X0[1: M + 1, 1: N + 1] = Y
    Y0 = np.zeros((M + 2, N + 2))
    Y0[1: M + 1, 1: N + 1] = Y
    X = np.zeros((M + 2, N + 2))
    Zx = np.zeros((M + 2, N + 2))
    Zy = np.zeros((M + 2, N + 2))
    Ux = np.zeros((M + 2, N + 2))
    Uy = np.zeros((M + 2, N + 2))
    K = 0
    lamda, rho = para.regularization, para.admmregularization
    num, err = para.most_iter_num, para.convergence
    alpha_ratio = para.nonconvexity_ratio
    
    while K < num and np.linalg.norm(X - X0, 2) > err:
        # update X
        X0 = X
        RHS = Y0 + lamda * rho*(diff.Dxt(Zx) + diff.Dyt(Zy)) - lamda * (diff.Dxt(Ux) + diff.Dyt(Uy))
        X = np.zeros((M + 2, N + 2))
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                X[i,j] = ((X0[i + 1, j] + X0[i - 1, j] + X0[i, j + 1] + X0[i, j - 1]) * lamda * rho 
                    + RHS[i, j]) / (1 + 4 * lamda * rho)
                
        # update Z
        Tx = Ux/rho + diff.Dx(X)
        Ty = Uy/rho + diff.Dy(X)
        Zx = shrink_mctv(Tx, 1/rho, alpha_ratio, num, err * 10)
        Zy = shrink_mctv(Ty, 1/rho, alpha_ratio, num, err * 10)
        
        # update U
        Ux = Ux + (diff.Dx(X) - Zx)
        Uy = Uy + (diff.Dy(X) - Zy)
        
        K += 1

    return X[1: M + 1, 1: N + 1]

def shrink(Y, lamda):
    return np.fmax(np.fabs(Y) - lamda, 0) * np.sign(Y)

def shrink_mctv(Y, lamda, alpha_ratio, num, err):
    
    M, N = np.shape(Y)
    X = np.zeros((M, N))
    U = np.ones((M, N))
    K = 0
    alpha = alpha_ratio / lamda
    
    while K < num and np.linalg.norm(X - U, 2) > err:
        
        U = X
        W = Y + lamda * alpha * (X - shrink(X, 1 / alpha))
        X = shrink(W, lamda)
        K += 1
    
    return X