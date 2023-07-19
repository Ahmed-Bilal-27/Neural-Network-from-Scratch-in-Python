# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:46:22 2023

@author: Bilal
"""

inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [ 0.2 , 0.8 , - 0.5 , 1.0 ]
bias = 2.0
output = (inputs[ 0 ] * weights[ 0 ] +
inputs[ 1 ] * weights[ 1 ] +
inputs[ 2 ] * weights[ 2 ] +
inputs[ 3 ] * weights[ 3 ] + bias)
print (output)