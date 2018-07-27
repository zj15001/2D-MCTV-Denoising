# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:05:23 2018

@author: Yilin Liu
"""

class Parameter(object):
    
    def __init__(self, re, ad, it, co, no):
        self.regularization = re
        self.admmregularization = ad
        self.most_iter_num = it
        self.convergence = co
        self.nonconvexity_ratio = no
    
    def print_value(self):
        print(self.regularization, self.admmregularization, 
                  self.most_iter_num, self.convergence, self.nonconvexity_ratio)