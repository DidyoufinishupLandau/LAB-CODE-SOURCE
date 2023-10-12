# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pattern_generator
from hadmard_mask import *

image = np.zeros((128, 128), dtype=np.uint8)
circle_center = (20, 20)
circle_radius = 20
for i in range(32):
    for j in range(32):
        if (i - circle_center[0]) ** 2 + (j - circle_center[1]) ** 2 <= circle_radius ** 2:
            image[i, j] = 1


def random_pattern(width, height, sparsity):
    mask_array = [(np.random.rand(height, width) < sparsity).astype(int) for i in range(width * height)]
    return mask_array


def reconstruction_image(pattern_array, intensity_array):
    weighting_matrix = np.sum(pattern_array, axis=0)
    image_list = []
    for i in range(len(pattern_array)):
        image_list.append(intensity_array[i] * pattern_array[i])
    image_array = np.array(image_list)
    image_matrix = np.sum(image_array, axis=0) / weighting_matrix
    return image_matrix


def plot_pixel(image_matrix):
    plt.imshow(image_matrix, cmap="gray")
    plt.show()


class SimulationCompressingImage:
    def __init__(self, simulation_image, light_intensity, width, height):
        self.light_intensity = light_intensity
        self.width = width
        self.height = height
        self.num_pixel = width * height
        self.simulation_image = simulation_image

    def simulate_random_pattern(self, sparsity=0.50, sampling_rate=1, noise_rate=0):
        assert sampling_rate <= 1, "The sampling rate must less than one."
        random_mask_array = random_pattern(self.width, self.height, sparsity)
        reverse_pattern_array = [(random_mask_array[i] == 0).astype(int) for i in range(len(random_mask_array))]
        image = []

        for i in range(int(self.width * self.height * sampling_rate)):
            mask = random_mask_array[i]
            reverse_mask = reverse_pattern_array[i]
            fractional_signal = np.sum((mask == self.simulation_image).astype(int)) / self.num_pixel

            reverse_fractional_signal = np.sum((reverse_mask == self.simulation_image).astype(int)) / self.num_pixel
            photo_diode_signal = self.light_intensity * fractional_signal
            photo_diode_reverse_signal = self.light_intensity * reverse_fractional_signal

            # signal_noise = np.random.randint(-100,100,[self.width, self.height])/100*noise_rate*self.light_intensity
            signal_noise = np.random.randint(-100, 100) / 100 * noise_rate * self.light_intensity
            reverse_signal_nose = np.random.randint(-100, 100) / 100 * noise_rate * self.light_intensity

            signal = photo_diode_signal + signal_noise
            reverse_signal = photo_diode_reverse_signal + reverse_signal_nose

            image.append(signal * mask + reverse_signal * reverse_mask)
        image = np.sum(np.array(image), axis=0)
        return image

    def simulate_hadmard_pattern(self, sampling_rate=1, noise_rate=0):
        matrix_ones, matrix_negative_ones = walsh_matrix(self.width ** 2)
        mask_matrix_one = walsh_to_hadmard_mask(matrix_ones)
        mask_negative_ones = walsh_to_hadmard_mask(matrix_negative_ones)
        image = []
        for i in range(int(len(mask_matrix_one) * sampling_rate)):
            mask = mask_matrix_one[i]
            reverse_mask = mask_negative_ones[i]

            fractional_signal = np.sum((mask == self.simulation_image).astype(int)) / self.num_pixel
            reverse_fractional_signal = np.sum((reverse_mask == self.simulation_image).astype(int)) / self.num_pixel

            photo_diode_signal = self.light_intensity * fractional_signal
            photo_diode_reverse_signal = self.light_intensity * reverse_fractional_signal

            # signal_noise = np.random.randint(-100,100,[self.width, self.height])/100*noise_rate*self.light_intensity
            signal_noise = np.random.randint(0, 100) / 100 * noise_rate * self.light_intensity
            reverse_signal_nose = np.random.randint(0, 100) / 100 * noise_rate * self.light_intensity

            signal = photo_diode_signal + signal_noise
            reverse_signal = photo_diode_reverse_signal + reverse_signal_nose
            image.append(signal * mask + reverse_signal * reverse_mask)

        image = np.sum(np.array(image), axis=0)
        return image


s = SimulationCompressingImage(image, 1, 128, 128)
simulated_image = s.simulate_hadmard_pattern(sampling_rate=1)
plot_pixel(image)
plot_pixel(simulated_image)




