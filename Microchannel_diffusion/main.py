from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from scipy.special import erf
from scipy.optimize import curve_fit

def Δt(l):
    hight_of_channel = 75 #micrometers
    width_of_channel = 500 #micrometers
    flow_rate = 80e9/60 # um^3/sek

    return l * hight_of_channel*width_of_channel/flow_rate

def list_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def main():
    lengsth_pixels = np.array([814.1, 1222.194, 1627.8, 2177.831])
    lengsth_pixels = [53.7, 80.62, 107.37, 143.656]
    folder_path = "data/Mixing/Gray_scale_data_junction"
    line_paths = list_all_files(folder_path)
    microns_per_pixel = 0.55
    
    result_array = []
    x_list = []
    Diffusion_constants = []

    for path, length in zip(line_paths, lengsth_pixels):
        data = pd.read_csv(path)

        xs = data['Distance_(pixels)']
        xs = xs/max(xs)                 #Noramlisation
        mean = xs[xs.size -1 ]/2
        xs = xs - mean
        x_list.append(xs)
        
        intensity = data['Gray_Value']
        initial = intensity[0]
        c_prop = np.log(initial/intensity)
        c_prop = c_prop/min(c_prop)  # Noramlisation 

        print(length)
        t = Δt(length)
        def analytical(x,D,c0):
            return -c0*0.5*(1-erf((x)/(np.sqrt(4*D*t)))) + 1
        
        param, param_cov = curve_fit(analytical , xs, c_prop)
        D, c0 = param
        
        Diffusion_constants.append(D)

        ans = analytical(xs, D, c0)
        result_array.append(ans)

        plt.scatter(xs,c_prop)
        plt.plot(xs,ans, color="r")
        plt.show()
    

    plt.figure()
    for results, D, x in zip(result_array, Diffusion_constants, x_list):
        print(D)
        plt.plot(x, results, label=f'D = {round(D,5)}') 

    plt.legend(loc =  'upper left')                
    plt.show()
    print("avrage D = ",np.mean(Diffusion_constants) , " std = ", np.std(Diffusion_constants))

if __name__ == "__main__":
    main()

