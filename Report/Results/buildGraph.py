# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

def map_steps(min=5, max=50, stepSize=5):
    nSteps = int((max - min) / stepSize )

    mapSteps = None

    for step in range(0, nSteps):
        step_min = min + stepSize * step
        step_max = step_min + stepSize
        newStep = np.array((step, step_min, step_max))

        if mapSteps is None:
            mapSteps = newStep
        else:
            mapSteps = np.vstack((mapSteps, newStep))

    return nSteps, mapSteps

def plot_offset(np_camera_params, np_51_50epochs, np_51_SSIM, np_71_40epochs, np_davis):
    
    
    #-------------------------------------------------------------
    #Offset 
    (nSteps, mapSteps) = map_steps(0.05, 0.50, 0.05)
    
    run_to_step = dict()
    
    # returns mapping from runNumber to offset step number according to mapping
    runnumber_column_index = 0
    offset_column_index = 8
    step_column_index = 0
    min_column_index = 1
    max_column_index = 2
    
    # find rows in np_camera_params that correspond to the step
    for step in mapSteps:
        for i in range(137):  #len(np_camera_params)):
            if np_camera_params[i][0][offset_column_index] >= step[min_column_index] and np_camera_params[i][0][offset_column_index] <= step[max_column_index]:
                run_to_step[int(np_camera_params[i][0][runnumber_column_index])] = int(step[step_column_index])
    
    # Offset - prepare graph
    index = np.arange(len(mapSteps))
    labels = []
    min_column_index = 1
    max_column_index = 2
    
    # x-axis labels
    for j, step in enumerate(mapSteps):
        labels.append("{:0.2f} - {:0.2f}".format(step[min_column_index], step[max_column_index]))
    
    plt.figure(1, figsize=(10, 8),)
    plt.xlabel('Offset(m)', fontsize=9)
    plt.ylabel('SSIM', fontsize=9)
    plt.xticks(index, labels, fontsize=9, rotation=30)
    plt.title('SSIM depending on the offset')
    
    plt.figure(2, figsize=(10, 8),)
    plt.xlabel('Offset(m)', fontsize=9)
    plt.ylabel('PSNR', fontsize=9)
    plt.xticks(index, labels, fontsize=9, rotation=30)
    plt.title('PSNR depending on the offset')
    
    for file_num in range(4):
        
        if file_num == 0:
            results = np_51_50epochs
        elif file_num == 1:
            results = np_51_SSIM
        elif file_num == 2:
            results = np_71_40epochs
        else:
            results = np_davis
    
        list_ssim = []
        list_psnr = []
        list_fn_ratio = []
        list_fp_ratio = []
        
        num_columns = results.shape[1]
        
        if num_columns == 9:  # silhouette included
            # calculate ratios
            fn_ratio = results[:, 7] / results[:, 6]
            fp_ratio = results[:, 8] / results[:, 6]
        
            # Add 3 columns - true pixels, fn, fp
            results = np.c_[results, fn_ratio]
            results = np.c_[results, fp_ratio]
            num_columns = 11
        
        for step in range(len(mapSteps)):
        
            total_ssim = 0
            total_psnr = 0
            total_fn_ratio = 0
            total_fp_ratio = 0
        
            count = 0
        
            for runNumber in run_to_step:
                if run_to_step.get(runNumber) == step:
        
                    selected_rows = None
                    if num_columns == 4:
                        selected_rows = results[results[:,0] == runNumber ][:, np.array([False, False, True, True])]
                    elif num_columns == 5:
                        selected_rows = results[results[:, 0] == runNumber][:, np.array([False, False, True, True, False])]
                    elif num_columns == 6:
                        selected_rows = results[results[:, 0] == runNumber][:, np.array([False, False, True, True, False, False])]
                    elif num_columns == 11:
                        selected_rows = results[results[:, 0] == runNumber][:, np.array([False, False, True, True, False, False, False, False, False, True, True])]
        
                    total_ssim += sum (selected_rows[:, 0])
                    total_psnr += sum (selected_rows[:, 1])
        
                    if num_columns == 11:
                        total_fn_ratio += sum (selected_rows[:, 2])
                        total_fp_ratio += sum(selected_rows[:, 3])
        
                    count += len(selected_rows)
        
            avg_ssim = total_ssim / count
            avg_psnr = total_psnr / count
            avg_fn_ratio = 0
            avg_fp_ratio = 0
        
            if num_columns == 11:
                avg_fn_ratio = total_fn_ratio / count
                avg_fp_ratio = total_fp_ratio / count
        
            print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr} for step {step}')
        
            list_ssim.append(avg_ssim)
            list_psnr.append(avg_psnr)
        
            if num_columns == 11:
                list_fn_ratio.append(avg_fn_ratio)
                list_fp_ratio.append(avg_fp_ratio)
    
        # plot ssim       
        plt.figure(1)
        plt.plot(index, list_ssim)       
        
        plt.figure(2)
        plt.plot(index, list_psnr)       
        
            #for i, v in zip(index, list_ssim):
            #    plt.text(i, v, "{:0.4f}".format(v), fontsize=7)
    
    plt.figure(1)
    plt.legend(['y = MV 51 L1 50 epochs', 'y = MV 51 50 L1 + 10 SSIM epochs', 'y = MV 71 L1 40 epochs', 'y = Non-MV Davis 100 epochs'], loc='lower left')       
    #plt.show()
    #plt.savefig(join(results_folder, "SSIM_Offset_combined.png"))
    
    plt.figure(2)
    plt.legend(['y = MV 51 L1 50 epochs', 'y = MV 51 50 L1 + 10 SSIM epochs', 'y = MV 71 L1 40 epochs', 'y = Non-MV Davis 100 epochs'], loc='upper right')       
    #plt.show()
    #plt.savefig(join(results_folder, "PSNR_Offset_combined.png"))

results_folder = "U:\Dissertation\Literature-Review\Report\Results"

file_camera_params = join(results_folder,"cameraParams_8.npy")
file_frames = join(results_folder,"per_frame_8.npy")
    
file_51_50epochs = join (results_folder, "51_50epochs", "ssim_psnr_all_8_significant_silhouettes.npy" )
file_51_SSIM = join (results_folder, "51_SSIM_10epochs", "ssim_psnr_all_8_significant_silhouettes.npy" )
file_davis = join (results_folder, "DAVIS_100epochs", "ssim_psnr_all_8_significant_silhouettes.npy" )
file_71_40epochs = join (results_folder, "71_40epochs", "ssim_psnr_all_8_significant_distance_from_center.npy" )

np_camera_params = np.load(file_camera_params)
np_frames = np.load(file_frames)

np_51_50epochs = np.load(file_51_50epochs)
np_51_SSIM = np.load(file_51_SSIM)
np_davis = np.load(file_davis)
np_71_40epochs = np.load(file_71_40epochs)

plot_offset(np_camera_params, np_51_50epochs, np_51_SSIM, np_71_40epochs, np_davis)
