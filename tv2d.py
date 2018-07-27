# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:01:12 2018

@author: Yilin Liu

Reference: Yang, S., Wang, J., Fan, W., Zhang, X., Wonka, P. & Ye, J. 
           An Efficient ADMM Algorithm for Multidimensional Anisotropic 
           Total Variation Regularization Problems. 
           Proceedings of the 19th ACM SIGKDD International Conference on 
           Knowledge Discovery and Data Mining - KDD’13 (2013). 
           doi: 10.1145/2487575.2487586

Algorithm for arg_min_X 0.5|Y - X|_2^2 + lamda*|X|_TV (|X|_TV = |DX|_1) (ADMM)
"""

import numpy as np
import diff

def denoising_2D_TV(Y, para):
    
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
    
    while K < num and np.linalg.norm(X - X0, 2) > err:
        # update X
        X0 = X
        RHS = Y0 + lamda * rho*(diff.Dxt(Zx) + diff.Dyt(Zy)) - lamda * (diff.Dxt(Ux) + diff.Dyt(Uy))
        X = np.zeros((M + 2, N + 2))
        
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                X[i, j] = ((X0[i + 1, j] + X0[i - 1, j] + X0[i, j + 1] + X0[i, j - 1]) * lamda * rho 
                    + RHS[i, j]) / (1 + 4 * lamda * rho)
                
        # update Z
        Tx = Ux/rho + diff.Dx(X)
        Ty = Uy/rho + diff.Dy(X)
        Zx = shrink(Tx, 1/rho)
        Zy = shrink(Ty, 1/rho)
        
        # update U
        Ux = Ux + (diff.Dx(X) - Zx)
        Uy = Uy + (diff.Dy(X) - Zy)
        
        K += 1

    return X[1: M + 1, 1: N + 1]

def shrink(Y, lamda):
    return np.fmax(np.fabs(Y) - lamda, 0) * np.sign(Y)