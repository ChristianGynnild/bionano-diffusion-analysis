from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from scipy.special import erf
from scipy.optimize import curve_fit

def Δt(l):
    hight_of_channel = 75 #micrometers
    width_of_channel = 500 #micrometers
    flow_rate = 40e9/60 # um^3/sek

    return l * hight_of_channel*width_of_channel/flow_rate

def list_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def main():
    lengsth_pixels = np.array([845.6, 917.812, 959.699, 1022.984]) #This is in mu meters from the software
    folder_path = "data/Mixing/Gray_scale_data_junction"
    line_paths = list_all_files(folder_path)
    
    result_array = []
    x_list = []
    Diffusion_constants = []

    for path, length in zip(line_paths, lengsth_pixels):
        data = pd.read_csv(path)

        xs = data['Distance_(unit)']
        max_x = max(xs)
        xs = xs/max_x                 #Noramlisation
        mean = xs[xs.size -1 ]/2
        xs = xs - mean
        x_list.append(xs)
        
        intensity = data['Gray_Value']
        initial = intensity[0]
        c_prop = -np.log(initial/intensity)
        c_prop -= np.mean(c_prop[(max(c_prop.index) -1)*0.2 > c_prop.index])
        c_prop = c_prop/np.mean(c_prop[(max(c_prop.index) -1)*0.8 < c_prop.index])  # Noramlisation 

        t = Δt(length)
        def analytical(x,D,):
            return -0.5*(1-erf((x)/(np.sqrt(4*D*t)))) + 1
        
        param, param_cov = curve_fit(analytical , xs, c_prop)
        D = param[0]
        
        Diffusion_constants.append(D*max_x**2)

        ans = analytical(xs, D)
        result_array.append(ans)

        plt.scatter(xs,c_prop)
        plt.plot(xs,ans, color="r")
        plt.savefig(f"./Microchannel_diffusion/Plots/plot_{path.split("/")[-1].split("\\")[-1].split(".")[0]}.jpg")
        plt.show()
    

    plt.figure()
    for results, D, x in zip(result_array, Diffusion_constants, x_list):
        print(D)
        plt.plot(x, results, label=f'D = {round(D,5)}') 

    plt.legend(loc =  'upper left')                
    plt.savefig("./Microchannel_diffusion/Plots/ALL_D.jpg")
    plt.show()
    print("Diffusion constant: ")
    print(Diffusion_constants)
    print("Mean D = ",np.mean(Diffusion_constants) , " std = ", np.std(Diffusion_constants))
    k_b = 1.3806503 * 10**(-23)
    nu = 1.002 * 10**(-3)              # Pas at 20*C
    T = 20+273
    radius_func = lambda d: k_b*T/(6*np.pi*nu*(d*10**(-12)))
    Diffusion_constants = np.array(Diffusion_constants)
    radius = radius_func(Diffusion_constants)
    print("Hydrodynamic radius: ")
    print(radius)
    print("Mean R = ", np.mean(radius), "std = ", np.std(radius))
    

if __name__ == "__main__":
    main()

