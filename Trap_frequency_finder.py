# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:03:00 2023

@author: Joe Bloomer
"""
#Import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy.constants as const
from numpy.fft import fft, fftfreq, fftshift
from scipy.optimize import curve_fit
import re

# Define Yb polarizability at 1064 nm in SI units
yb_pol_1064 = 159.56539113196172 * (4*np.pi * const.epsilon_0) * const.physical_constants['Bohr radius'][0]**3


def yb_inten2pot(I,yb_pol):
    # Function converts an intensity in SI units to a dipole potential in SI units for a certain polarizability
    return -1/(2*const.epsilon_0*const.c) * yb_pol * I


def quadratic(x,x0,w,U_0):
    # Function for fitting a quadratic to an approximate harmonic potential
    return -U_0*(1 - 2*((x-x0)/w)**2)


def trap_freq(w_0,U_0,m_yb):
    return np.sqrt(4*U_0 / (m_yb * w_0**2))/(2*np.pi)

def load_files_from_specific_folders(root_folder, folder_format,m_yb = const.m_p*174, ylim = 1.5e-4, N = 1501,dy = 2*1.5e-4/1501, norm = 2, Ignore_fitting_error = True):
    # Get a list of all folders that match the specified format
    folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f)) and f.endswith(folder_format)]
    
    print(folders)
    
    gal_angle = np.zeros(len(folders))
    
    av_tot_power = np.zeros(len(folders))
    err_tot_power = np.zeros(len(folders))
    
    av_spacings = np.zeros(len(folders))
    err_spacings = np.zeros(len(folders))
    
    av_trap_freq = np.zeros(len(folders))
    err_trap_freq = np.zeros(len(folders))
    
    av_trap_depth = np.zeros(len(folders))
    err_trap_depth = np.zeros(len(folders))
    
    av_fringe_centre = np.zeros(len(folders))
    err_fringe_centre = np.zeros(len(folders))
    
    
    # Iterate through each folder
    
    for j,folder in enumerate(folders):
        print(f"Open Folder: {folder}", f"Folder Number: {j}")
        folder_path = os.path.join(root_folder, folder)
        gal_angle[j] = float(re.findall(r"[+-]?\d+(?:\.\d+)?",folder)[0])
        print(gal_angle[j])
        
        # Get a list of all files in the folder using glob
        files = glob.glob(os.path.join(folder_path, '*'))
        
        fringe_centre = np.zeros(len(files))
        trap_freq_file = np.zeros(len(files))
        U_0 = np.zeros(len(files))
        tot_power = np.zeros(len(files))
        
        spacings = np.zeros(len(files))
        
        # Process each file
        for i, file in enumerate(files):
            # Here, you can add your code to load or process the files as needed
            print(f"Loading file: {file}",f"File number: {i}")
            y = np.arange(-ylim,ylim,dy)
            data = np.loadtxt(file,encoding='utf-16',skiprows=24,usecols = np.arange(1,1501,1))*1e4
            data = data#*norm/(np.sum(data)*dy**2)
            
            tot_power[i] = np.sum(data)*dy**2
            print(np.sum(data)*dy**2)
            
            data_cent = data[:,int(N/2)]
            norm_cent = np.sum(data_cent)
            data_int = np.sum(data,axis=1)
            data_int = data_int*norm_cent/np.sum(data_int)
                   
            w = fftfreq(N,dy)
            w = fftshift(w)
            
            data_int_fft = fftshift(fft(data_int))
            data_int_fft_abs = np.abs(data_int_fft)
            
            cut = 755
            wc = w[cut:]
            
            Amp_max = np.argmax(data_int_fft_abs[cut:])
            spacings[i] = 1/wc[Amp_max]
            
            plt.figure()
            plt.plot(w,data_int_fft_abs)
            plt.scatter(wc[Amp_max],data_int_fft_abs[cut:][Amp_max])
            
            plt.show()
            
            plt.figure()
            plt.plot(y,data_cent)
            plt.plot(y,data_int)
            plt.show()
            
            potentials = yb_inten2pot(data_int)
            
            print(np.min(potentials/const.Boltzmann*1e6))
            
            plt.figure()
        
            plt.plot(y,potentials/const.Boltzmann*1e6)
            
            plt.show()
            
            Range = spacings[i]/4
           
            print(Range)
            indices = np.where(y < Range)[0]
            indices = np.where(y[indices] > -Range)[0]
            print(indices)
            #indices = np.arange(498,505)
            y_harmonic = y[indices]
            potentials_harmonic = potentials[indices]
            
           
            Min_pot_y = y_harmonic[np.argmin(potentials_harmonic)]
            Min_pot = potentials_harmonic[np.argmin(potentials_harmonic)]
            
            indices_new = np.where(y < Range + Min_pot_y)[0]
            indices_new = np.where(y[indices_new] < -Range + Min_pot_y)[0]
            
            y_harmonic_new = y[indices_new]
            potentials_harmonic_new = potentials[indices_new]
            

            Min_pot_y_new = y_harmonic_new[np.argmin(potentials_harmonic_new)]
            Min_pot_new = potentials_harmonic_new[np.argmin(potentials_harmonic_new)]

            plt.plot(y*1e6,potentials/const.Boltzmann*1e6)
            
            
            if Ignore_fitting_error == True:
                try:
                    popt, covt = curve_fit(quadratic, y_harmonic_new, potentials_harmonic_new, p0 = [Min_pot_y_new,spacings[i],Min_pot_new])
                except:
                    pass
            else:
                popt, covt = curve_fit(quadratic, y_harmonic_new, potentials_harmonic_new, p0 = [Min_pot_y_new,spacings[i],Min_pot_new])
                
                
                
                
                
            print(popt)
            
            #print(np.sqrt(popt[0]*2/m_yb)/(2*np.pi))
            
            fringe_centre[i] = popt[0]
            w_0 = np.abs(popt[1])
            
            U_0[i] = popt[2]
            U_0i = popt[2]
            trap_freq_file[i] = trap_freq(w_0, U_0i, m_yb)
            print(U_0[i]/const.Boltzmann*1e6)
            plt.figure()
        
            plt.plot(y_harmonic_new*1e6,potentials_harmonic_new/const.Boltzmann*1e6)
            plt.plot(y*1e6,quadratic(y, *popt)/const.Boltzmann*1e6)
            plt.ylim(-95,5)
            plt.xlim(-5,5)
            plt.xlabel('Detector Position ($\mu$m)')
            plt.ylabel('Potential (\muK)')
            
            plt.show()
        
        av_tot_power[j] = np.average(tot_power)
        err_tot_power[j] = np.std(tot_power,ddof=1)/np.sqrt(len(files))
        
        av_spacings[j] = np.average(spacings)
        err_spacings[j] = np.std(spacings,ddof=1)/np.sqrt(len(files))
        
        av_trap_freq[j] = np.average(trap_freq_file)
        err_trap_freq[j] = np.std(trap_freq_file,ddof=1)/np.sqrt(len(files))
        
        av_trap_depth[j] = np.average(U_0)
        err_trap_depth[j] = np.std(U_0,ddof = 1)/np.sqrt(len(files))
        
        av_fringe_centre[j] = np.average(fringe_centre)
        err_fringe_centre[j] = np.std(fringe_centre,ddof = 1)/np.sqrt(len(files))
        
    return gal_angle, av_tot_power, err_tot_power, av_spacings, err_spacings, av_trap_depth, err_trap_depth, av_trap_freq, err_trap_freq, av_fringe_centre, err_fringe_centre
         

# Specify the root folder you want to start from
#root_folder = 'C:/Users/Joe Bloomer/OneDrive - Durham University/PhD/Accordion Lattice/Accordion Lattice/Zemax setup/Sensitivities_new/Focus_lens/Focus_xtilt_-0.2deg'
root_folder = 'C:/Users/Joe Bloomer/OneDrive - Durham University/PhD/Accordion Lattice/Accordion Lattice/Zemax setup/No Misalignments'

# Specify the format of the folders you want to process
folder_format = "deg"

# Call the function to load files from specific folders
gal_angles,  av_tot_power, err_tot_power, av_spacings, err_spacings, av_trap_depth, err_trap_depth, av_trap_freq, err_trap_freq, av_fringe_centre, err_fringe_centre = load_files_from_specific_folders(root_folder, folder_format,Ignore_fitting_error=False)
#%%
# Plot the average total power with galvo angle
plt.figure()

plt.errorbar(gal_angles, av_tot_power, yerr = err_tot_power, marker = 'o',c = 'black', linestyle='', elinewidth=1, capsize = 1)

plt.xlabel('Galvo angle ($^\circ$)')
plt.ylabel('Total Power (W)')

plt.show()

#%%
# Plot the average lattice spacings with galvo angle

plt.figure()

plt.errorbar(gal_angles, av_spacings*1e6, yerr = err_spacings*1e6, marker = 'o',c = 'black',linestyle='',elinewidth=1,capsize = 1)

plt.xlabel('Galvo angle ($^\circ$)')
plt.ylabel('Lattice Spacing ($\mu$m)')

plt.show()

#%%
plt.figure()
plt.errorbar(gal_angles, av_trap_freq, yerr = err_trap_freq,marker = 'o',c = 'black',linestyle='',elinewidth=1,capsize = 1)
plt.xlabel('Galvo angle ($^\circ$)')
plt.ylabel('Trap Frequency (Hz)')
plt.show()

#%%
plt.figure()
plt.errorbar(gal_angles, av_fringe_centre*1e6, yerr = err_fringe_centre*1e6,marker = 'o',c = 'black',linestyle='',elinewidth=1,capsize = 1)
plt.xlabel('Galvo angle ($^\circ$)')
plt.ylabel('Fringe centre ($\mu m$)')
plt.show()

#%%

plt.figure()
plt.errorbar(gal_angles, av_trap_depth/const.Boltzmann*1e6, err_trap_depth/const.Boltzmann*1e6,marker = 'o',elinewidth = 1, linewidth = 0)
plt.xlabel('Galvo angle ($^\circ$)')
plt.ylabel('Maximum Trap Depth ($\mu$K)')
plt.show()
