# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

def bit_reverse_permutation(num_bits):
    data = np.linspace(0,2**num_bits-1, 2**num_bits)
    n = len(data)
    num_bits = len(bin(n - 1)) - 2
    result = [0] * n
    for i in range(n):
        reversed_index = int(format(i, f'0{num_bits}b')[::-1], 2)
        result[reversed_index] = data[i]
    
    return result
def generate_gray_code(n):
    if n <= 0:
        return [""]
    smaller_gray_codes = generate_gray_code(n - 1)
    result = []
    for code in smaller_gray_codes:
        result.append("0" + code)
    for code in reversed(smaller_gray_codes):
        result.append("1" + code)

    return result

def gray_code_permutation(num_bits):
    gray_codes = generate_gray_code(num_bits)
    decimal_permutation = [int(code, 2) for code in gray_codes]
    return decimal_permutation

def walsh_matrix(system_size):
    num_bits = int(np.log2(system_size))
    hadmard_matrix = scipy.linalg.hadamard(system_size)
    hadamard_row = []
    for i in range(len(hadmard_matrix)):
        hadamard_row.append(hadmard_matrix[i])
    temp_walsh_row = []
    
    reverse_bit_string = bit_reverse_permutation(num_bits)
    for i in range(len(reverse_bit_string)):
        temp_walsh_row.append(hadamard_row[int(reverse_bit_string[i])])
    print("complete")
    walsh_matrix = np.zeros((0,system_size))
    gray_code_string = generate_gray_code(num_bits)
    for i in range(len(gray_code_string)):
        gray_code_string[i] = int(gray_code_string[i], 2)
    for i in range(len(gray_code_string)):
        walsh_matrix = np.vstack((walsh_matrix, temp_walsh_row[int(gray_code_string[i])]))
    print("complete")
    array_one = (walsh_matrix == 1).astype(int)
    array_negative_one = (walsh_matrix == -1).astype(int)
    return array_one, array_negative_one


def walsh_to_hadmard_mask(input_matrix):
    small_matrix_size = int(np.sqrt(len(input_matrix[0])))
    num_rows, num_cols = input_matrix.shape
    num_small_matrices = num_rows // small_matrix_size
    small_matrices = []
    
    for i in range(num_small_matrices):
        for j in range(num_small_matrices):
            start_row = i * small_matrix_size
            end_row = start_row + small_matrix_size
            start_col = j * small_matrix_size
            end_col = start_col + small_matrix_size
            print(start_row, end_row)
            small_matrix = input_matrix[start_row:end_row, start_col:end_col]
            small_matrices.append(small_matrix)
    return np.array(small_matrices)

def walsh_matrix(system_size):
    hadmard_matrix = scipy.linalg.hadamard(system_size)
    array_one = (hadmard_matrix == 1).astype(int)
    array_negative_one = (hadmard_matrix == -1).astype(int)
    return array_one, array_negative_one
def walsh_to_hadmard_mask(input_matrix):
    small_matrix_size = int(np.sqrt(len(input_matrix[0])))
    num_rows, num_cols = input_matrix.shape
    num_small_matrices = num_rows // small_matrix_size
    small_matrices = []

    reverse_bit_string = bit_reverse_permutation(int(np.log2(num_rows)))
    gray_code_string = generate_gray_code(int(np.log2(num_rows)))
    for i in range(len(gray_code_string)):
        gray_code_string[i] = int(gray_code_string[i], 2)

    def mapping(n):
        n = gray_code_string[int(reverse_bit_string[n])]
        return n
    mapping_list = [mapping(i) for i in range(num_rows)]
    mapping_list = np.array(mapping_list)
    new_list = []
    for i in range(len(mapping_list)):
        new_list.append(np.where(mapping_list==i)[0][0])
    new_list = np.array(new_list)
    for i in range(num_small_matrices):
        for j in range(num_small_matrices):
            start_row = i * small_matrix_size
            end_row = start_row + small_matrix_size
            start_col = j * small_matrix_size
            end_col = start_col + small_matrix_size
            small_matrix = []
            row_number = np.linspace(start_row, end_row - 1, end_row-start_row).astype(int)
            for n in range(len(row_number)):
                small_matrix.append(input_matrix[new_list[row_number[n]], start_col:end_col])
            small_matrix = np.array(small_matrix)
            small_matrices.append(small_matrix)
    return np.array(small_matrices)

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

def save_3d_matrix_to_file(file_path, data):
    """
    Save a 3D matrix to a text file.

    Args:
        file_path (str): The path to the text file where the matrix will be saved.
        data (numpy.ndarray): The 3D matrix to be saved.
    """
    with open(file_path, "w") as file:
        for matrix_2d in data:
            for row in matrix_2d:
                file.write(" ".join(map(str, row)) + "\n")

def load_3d_matrix_from_file(file_path):
    """
    Load a 3D matrix from a text file.

    Args:
        file_path (str): The path to the text file containing the matrix.

    Returns:
        numpy.ndarray: The loaded 3D matrix.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        depth = len(lines)
        matrix_2d = np.array([list(map(float, line.split())) for line in lines[0].split("\n") if line])
        height, width = matrix_2d.shape

        data = np.empty((depth, height, width))

        for i, line in enumerate(lines):
            matrix_2d = np.array([list(map(float, line.split())) for line in line.split("\n") if line])
            data[i, :, :] = matrix_2d

    return data
