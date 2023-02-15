"""
Author Pan Zhang
Given a csv file with two columns and ; as delimiter, the code can read file 
and apply fast fourier transform to the data.

The coherence length and wavelength of the light source is calculated.
All plot will be saved in the same file.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import constants

DELIMITER = ','
PLOT_RANGE = 1.1
sample_rate = 10000
N = int(2 * sample_rate)
velocity = 0.001*10**-3

def read_file(file_name):
    combined_data = np.zeros((0, 2))
    try:
        for name in file_name:
            file_data = np.loadtxt(name, delimiter=DELIMITER, skiprows=3)
            combined_data = np.vstack((combined_data, file_data))
        return combined_data
    except ValueError:
        print('File data should be in the format [x value, y value].')
        sys.exit()
    except OSError:
        print(f'File {name} cannot be found.')
        sys.exit()

def fourier_transform(raw_data):
    N = len(raw_data)
    freq_data = fft(raw_data[:, 1]) / N
    y = 2 * np.abs(freq_data[:N//2])
    frequency = fftfreq(N, raw_data[1,0] - raw_data[0,0])
    return frequency[:N//2], y


def plot(data_array_1, data_array_2, file_name):
    fig = plt.figure(figsize=(10, 8))
    main_plot = fig.add_subplot(211)
    
    # Get the index of the maximum value
    max_index = np.argmax(data_array_1[:,1])
    # Find the frequency value corresponding to the maximum value
    max_frequency = data_array_1[max_index, 0]
    data_array_1 = np.delete(data_array_1, [max_index],axis=0)
    
    max_index = np.argmax(data_array_2[:,1])
    # Find the frequency value corresponding to the maximum value
    max_frequency = data_array_2[max_index, 0]
    data_array_2 = np.delete(data_array_2, [max_index],axis=0)
    # Set the x-axis limit based on the maximum frequency
    x_min = max_frequency - 1
    x_max = max_frequency + 5
    
    main_plot.set_xlim(x_min, x_max)
    main_plot.plot(data_array_1[:,0], data_array_1[:,1], '-b')
    main_plot.plot(data_array_2[:,0], data_array_2[:,1], 'r')
    main_plot.set_xlabel("Frequency (Hz)")
    main_plot.set_ylabel("Amplitude")
    main_plot.set_title("Fourier Transform")
    main_plot.grid(True)
    
    max_amp_1 = data_array_1[np.argmax(data_array_1[:,1])][1]
    max_freq_1 = data_array_1[np.argmax(data_array_1[:,1])][0]
    
    max_amp_2 = data_array_2[np.argmax(data_array_2[:,1])][1]
    max_freq_2 = data_array_2[np.argmax(data_array_2[:,1])][0]
    
    
    plt.annotate(f'Max Amp: {max_amp_1:.2f}\nFreq: {max_freq_1:.2f}Hz',
    xy=(max_freq_1, max_amp_1), xycoords='data',
    xytext=(-100, -100), textcoords='offset points',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3,rad=.2"))
    
    plt.annotate(f'Max Amp: {max_amp_2:.2f}\nFreq: {max_freq_2:.2f}Hz',
    xy=(max_freq_2, max_amp_2), xycoords='data',
    xytext=(-100, -100), textcoords='offset points',
    arrowprops=dict(arrowstyle="->",
    connectionstyle="arc3,rad=.2"))
    
    plt.savefig(f'{file_name}.png')
    return data_array_1[np.argmax(data_array_1[:,1])], data_array_2[np.argmax(data_array_2[:,1])]


def plot2(data_array, file_name):
    fig = plt.figure(figsize=(10, 8))
    plot_x_axis_range = np.linspace(np.amin(data_array[:, 0]) -
                                    np.mean(data_array[:, 0]) * PLOT_RANGE
                                    ,
                                    np.amax(data_array[:, 0]) +
                                    np.mean(data_array[:, 0]) * PLOT_RANGE
                                    ,
                                    1000)

    main_plot = fig.add_subplot(211)
    main_plot.plot(data_array[:,0], data_array[:,1])
    main_plot.set_xlim(0, np.amax(np.abs(data_array[:,0])))
    main_plot.set_xlabel("Time (s)")
    main_plot.set_ylabel("Amplitude (v)")
    main_plot.set_title("Raw data")
    plt.savefig(f'{file_name}.png')
    plt.show()
    
def reshaper(array_list):
    x, y = array_list[0], array_list[1]
    new_array = np.hstack((x, y))
    return new_array

def search_turning_point(data_array, valve):
    turning_point = []
    turning_point_value = []
    for i in range(1, len(data_array)-1):
        if (data_array[i-1]<valve and data_array[i+1]>valve):
            turning_point.append(i)
            turning_point_value.append(data_array[i])
        elif (data_array[i-1]>valve and data_array[i+1]<valve):
            turning_point.append(i)
            turning_point_value.append(data_array[i])
    return np.array(turning_point), np.array(turning_point_value)

num = 3
for i in range(2, num + 1):
    ####################name the picture
    raw_data_A_file_name =  f"{i}A"
    raw_data_B_file_name =  f"{i}B"
    fourier_transform_combine = f"fourier transform {i}"
    file_1 = [f'{i}A.csv']
    file_2 = [f'{i}B.csv']
    ########################### get the frequency
    data_1 = read_file(file_1)
    plot2(data_1, raw_data_A_file_name)
    data_2 = read_file(file_2)
    plot2(data_2, raw_data_B_file_name)
    
    
    frequency_1, y_1 = fourier_transform(data_1)
    frequency_2, y_2 = fourier_transform(data_2)
    
    
    new_array_1 = np.hstack((frequency_1.reshape(len(frequency_1),1), y_1.reshape(len(y_1),1)))
    new_array_2 = np.hstack((frequency_2.reshape(len(frequency_2),1), y_2.reshape(len(y_2),1)))
    (max_frequency_1, max_signa_1), (max_frequency_2, max_signa_2)  = plot(new_array_1, new_array_2, fourier_transform_combine)
    wavelength_1 = velocity/max_frequency_1
    wavelength_2 = velocity/max_frequency_2
    print((wavelength_1, max_signa_1), (wavelength_2, max_signa_2))
    
    ########################
    #calcualte range of wavelength the light emits
    half_height_1 = max_signa_1/2 #half height
    turning_index_1, turning_value_1 = search_turning_point(y_1, half_height_1) 
    del_frequency_1 = turning_value_1[-1]-turning_value_1[0]
    
    half_height_2 = max_signa_2/2
    turning_index_2, turning_value_2 = search_turning_point(y_2, half_height_2)
    del_frequency_2 = turning_value_2[-1]-turning_value_2[0]
    
    del_wavelength_1 = (velocity)/(max_frequency_1)**2 * del_frequency_1
    del_wavelength_2 = (velocity)/(max_frequency_2)**2 * del_frequency_2
    coherence_length_1 = wavelength_1**2/del_wavelength_1
    coherence_length_2 = wavelength_2**2/del_wavelength_2
    ######################
    print(coherence_length_1, coherence_length_2)
