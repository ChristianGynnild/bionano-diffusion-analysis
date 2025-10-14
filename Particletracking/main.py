from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pims
import trackpy as tp
import os
from numpy import pi
from sklearn.linear_model import LinearRegression

def generate_trajectories(path, number_of_frames=-1):
    folder_path = "./Particletracking/data"
    os.makedirs(folder_path, exist_ok=True)
    new_path = folder_path + "/" + path.split("/")[-1].split(".")[0] + "_data_" + str(number_of_frames)
    if os.path.isfile(new_path):
        ds = pd.read_csv(new_path)
        return ds
    else:
        frames = pims.as_gray(pims.open(path))
        Data_frame = tp.batch(frames[:number_of_frames], 11, minmass=20, processes=5);
        trajectories = tp.link(Data_frame, 10, memory=5)
        
        trajectories.to_csv(new_path)
    return trajectories


def filter_trajectories(trajectories):           
    trajectories = tp.filter_stubs(trajectories, 5)
    #Out commented code migth be usefull for the rapport could be put into an actuall function or and if statmwnt if plots are desired
    # print('Before:', trajectories['particle'].nunique())
    # print('After:', trajectories_1['particle'].nunique())
    # plt.figure()
    # tp.mass_size(trajectories_1.groupby('particle').mean())
    # tp.mass_size(trajectories_2.groupby('particle').mean())
    trajectories = trajectories[(trajectories['size'] > 3.3)]# & (trajectories['signal'] > 7)]
        
        #(trajectories['mass'] > 50) &
    #                            (trajectories['size'] < 2.6) & 
    #                            (trajectories['size'] > 0.3) &  
    #                            (trajectories['ecc'] < 0.3) &
    #                            (trajectories['signal'] > 130)

    # plt.figure()
    # tp.annotate(trajectories_2[trajectories_2['frame'] == 0], frames[0]);
    # tp.plot_traj(trajectories_2)
    # plt.show()
    return trajectories

def remove_drift(trajectories):
    d = tp.compute_drift(trajectories)
    ds = tp.subtract_drift(trajectories.copy(), d)
    return ds

visualise = False

#First time this was actually required
if __name__ == "__main__":

    filepaths = ["data/white paint/A/BF-A-W-deep-fried.avi",
                 "data/white paint/A/DF-A-W-2-improved-DF-deep-fried.avi",
                 "data/white paint/A/PHC_A_W-deep-fried.avi",
                 "data/white paint/B/BF-B-W-deep-fried.avi",
                 "data/white paint/B/DF-B-W-4-improved-DF-deep-fried.avi",
                 "data/white paint/B/PHC_B_W-deep-fried.avi"
                 ]

    diffusion_constants = []
    hydrodynamic_radiuses = []

    for filepath in filepaths:
        print("Generate trajectories")
        trajectories = generate_trajectories(filepath, 120)
        print("Filtering trajectories")
        trajectories = filter_trajectories(trajectories)
        print("Removing drift velocity")
        trajectories = remove_drift(trajectories)


        if visualise:  
            x_offset = 1
            y_offset = 1

            position = lambda x,y:(trajectories['x'] < x+x_offset) & (trajectories['x'] > x-x_offset) & \
                                (trajectories['y'] < y+y_offset) & (trajectories['y'] > y-y_offset)

            print(trajectories[position(423, 1133)])
            print(trajectories.head())


            frames = pims.as_gray(pims.open(filepath))

            for i in range(10):
                ax = tp.annotate(trajectories[trajectories["frame"] == i], frames[i], plot_style={"markersize": 2, "alpha": 0.2})

            #print("Hm")
            #figure = plt.figure()
            #figure.add_axes(ax)
            #figure.savefig("annotation.png")


        # tm = generate_datastructure("data/white paint/A/BF-A-W_2.avi", 120)
        #tp.plot_traj(trajectories)
        #plt.show()
        #plt.figure()
        
        if True:
            print("Finding mean square displacement")
            em = tp.emsd(trajectories, 0.151285759**2, 5) # microns per pixel = 0.55., frames per second = 122/19.99


            plt.figure()
            plt.plot(em.index, em.values)

            x = np.reshape(em.index, (len(em.index), 1))
            regression = LinearRegression(fit_intercept=False).fit(x, em.values)
            D = regression.coef_[0]/4

            print(f"Results for {filepath}:")

            print(f"D={round(D,4)} [(µm)^2/s]")

            plt.plot(em.index, regression.predict(x))
            plt.savefig("plot.png")

            k_B = 1.38*10**-23 # [J/K]
            T = 293 # [K]
            viscocity = 0.001 # [Pa*s]

            R = k_B*T/(6*pi*viscocity*D)*10**12*10**6
            print(f"R:{round(R,2)} [µm]")

            diffusion_constants.append(round(float(D), 4))
            hydrodynamic_radiuses.append(round(float(R),2))

        
    print(f"diffuion constants:{diffusion_constants} [(µm)^2/s]")
    print(f"mean: {round(np.mean(diffusion_constants), 4)} [(µm)^2/s]")
    print(f"std: {round(np.std(diffusion_constants), 4)} [(µm)^2/s]")


    print(f"hydrodynamic radius:{hydrodynamic_radiuses} [(µm)]")
    print(f"mean: {round(np.mean(hydrodynamic_radiuses), 2)} [(µm)]")
    print(f"std: {round(np.std(hydrodynamic_radiuses), 2)} [(µm)]")
    

        #x,y = em.index, em[em.index]
        #reg = LinearRegression().fit(x, y)


        #print(f"em:{em}")
        #em_2 = em[em.index < em.index[-1] / 1]
        #print(len(em_2.index), len(em.index))
        #fig, ax = plt.subplots()
        #ax.plot(em_2.index, em_2, 'o')
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
        #   xlabel='lag time $t$')

        #linear_regress = tp.utils.fit_powerlaw(em_2)  # performs linear best fit in log space, plots]
        #print(linear_regress)
        #print("D = ",linear_regress["A"]["msd"]/4)
