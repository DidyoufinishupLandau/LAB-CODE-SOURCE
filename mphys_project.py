# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:17:08 2023

@author: 43802
"""

import numpy as np
import matplotlib.pyplot as plt

def reconstruction_image(pattern_array, intensity_array):
    weighting_matrix = np.sum(pattern_array, axis=0)
    image_list = []
    for i in range(len(pattern_array)):
        image_list.append(intensity_array[i] * pattern_array[i])
    image_array = np.array(image_list)
    image_matrix = np.sum(image_array, axis=0)/weighting_matrix
    return image_matrix
def plot_pixel(image_matrix):
    plt.imshow(image_matrix, cmap="gray")
    plt.show()
pattern_array= []
for i in range(10):
    pattern_array.append(np.random.randint(0, 2, [8,8]))
pattern_array = np.array(pattern_array)
pattern_intensity = np.random.randint(0,10,[1,10])

image = reconstruction_image(pattern_array, pattern_intensity[0])
plot_pixel(image)