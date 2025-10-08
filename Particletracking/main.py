from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pims
import trackpy as tp
import os

def generate_datastructure(path, number_of_frames):
    folder_path = "./Particletracking/data"
    os.makedirs(folder_path, exist_ok=True)
    new_path = folder_path + "/" + path.split("/")[-1].split(".")[0] + "_data_" + str(number_of_frames)
    if os.path.isfile(new_path):
        ds = pd.read_csv(new_path)
        return ds
    else:
        frames = pims.as_gray(pims.open(path))
        Data_frame = tp.batch(frames[:number_of_frames], 11, minmass=20, processes=5);
        trajectories = tp.link(Data_frame, 5, memory=5)
        trajectories_1 = tp.filter_stubs(trajectories, 5)
        #Out commented code migth be usefull for the rapport could be put into an actuall function or and if statmwnt if plots are desired
        # print('Before:', trajectories['particle'].nunique())
        # print('After:', trajectories_1['particle'].nunique())
        # plt.figure()
        # tp.mass_size(trajectories_1.groupby('particle').mean())
        # tp.mass_size(trajectories_2.groupby('particle').mean())
        trajectories_2 = trajectories_1[ (trajectories_1['mass'] > 50) & (trajectories_1['size'] < 2.6) &  (trajectories_1['ecc'] < 0.3)]

        # plt.figure()
        # tp.annotate(trajectories_2[trajectories_2['frame'] == 0], frames[0]);
        # tp.plot_traj(trajectories_2)
        # plt.show()

    

        d = tp.compute_drift(trajectories_2)
        # d.plot()
        # plt.show()
        ds = tp.subtract_drift(trajectories_2.copy(), d)
        ds.to_csv(new_path)
        return ds

#First time this was actually required
if __name__ == "__main__":
    tm = generate_datastructure("./data/white paint/B/DF-B-W-modified.avi", 120)
    # tm = generate_datastructure("data/white paint/A/BF-A-W_2.avi", 120)
    tp.plot_traj(tm)
    plt.show()
    plt.figure()
    em = tp.emsd(tm, 0.55**2, 5) # microns per pixel = 0.55., frames per second = 122/19.99
    em_2 = em[em.index < em.index[-1] / 1]
    print(len(em_2.index), len(em.index))
    fig, ax = plt.subplots()
    ax.plot(em_2.index, em_2, 'o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')

    linear_regress = tp.utils.fit_powerlaw(em_2)  # performs linear best fit in log space, plots]
    print(linear_regress)
    print("D = ",linear_regress["A"]["msd"]/4)


