import segyio
from scipy.stats import ks_2samp
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from os.path import join as pjoin
import shutil
import os
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor


def depth_signal_checking(simulation_data, seismic_data):
    """
    Ensures that the depth (Z) values in both simulation and seismic datasets are positive.
    Input:
        simulation_data: numpy array
                         Simulation data with columns [X, Y, Z, Property].
        seismic_data: numpy array
                      Seismic data with columns [X, Y, Z, (Property)].
    Output:
        Tuple:
            - simulation_data: numpy array with positive Z values.
            - seismic_data: numpy array with positive Z values.
    """
    
    simulation_data[:, 2] = np.abs(simulation_data[:, 2])
    seismic_data[:, 2] = np.abs(seismic_data[:, 2])
    
    print("Done!")
    
    return simulation_data, seismic_data


def simulation_nan_treatment(simulation, value=0, method='replace'):
    """
    Treats NaN values in the simulation dataset by either replacing them or removing affected cells.
    Input:
        simulation: numpy array
                    Simulation data with shape (n_samples, n_features), last column is the property.
        value: numeric (default=0)
               Value to replace NaNs with when method='replace'.
        method: string ('replace' or 'remove')
                - 'replace': replaces NaNs in the property column with the specified value.
                - 'remove': removes rows where the property column is NaN.
    Output:
        simulation: numpy array
                    Updated simulation data after NaN treatment.
    """
    
    initial_samples = simulation.shape[0]
    
    if method == "replace":
        print(f"Method: {method}")
        print(f"Shape Prior to NaN treatment: {simulation.shape}")
        print(f"Prior to NaN treatment:\n{simulation}")
        simulation[:, -1] = np.where(np.isnan(simulation[:, -1]), value, simulation[:, -1])
        print(f"Shape After NaN treatment: {simulation.shape}")
        print(f"After NaN treatment:\n{simulation}")
    else:
        print(f"Method: {method}")
        print(f"Shape Prior to NaN treatment: {simulation.shape}")
        print(f"Prior to NaN treatment:\n{simulation}")
        simulation = simulation[np.isnan(simulation[:, -1]) != True]
        print(f"Shape After NaN treatment: {simulation.shape}")
        print(f"After NaN treatment:\n{simulation}")

    final_samples = simulation.shape[0]

    if initial_samples != final_samples:
        residual = initial_samples - final_samples
        print(f"WARNING!\n{residual} CELLS WERE REMOVED!!")
        
    print("Done!")        

    return simulation


def simulation_data_cleaning(simulation_data=None, value_to_clean=None):
    """
    Cleans the simulation dataset by removing samples with a specified unwanted value.
    Input:
        simulation_data: numpy array
                         Simulation data with columns [X, Y, Z, Property].
        value_to_clean: numeric
                        Value in the property column to be removed (e.g., -99.0 for invalid samples).
    Output:
        simulation_data: numpy array
                         Cleaned simulation data with specified values removed.
    """
    
    original_data = simulation_data.shape[0]
    print(f"Original number of samples in simulation model: {original_data}")
    
    # Filter out the unwanted value
    simulation_data = simulation_data[simulation_data[:, -1] != value_to_clean]
    print(f"Final number of samples after cleaning: {simulation_data.shape[0]}")
    
    # Calculate and report the percentage of data removed
    percentage_loss = ((original_data - simulation_data.shape[0]) / original_data) * 100
    print(f"Percentage loss: {round(percentage_loss, 2)}%")
    
    print("Done!")
    
    return simulation_data
	

def plot_simulation_distribution(sim_array_xyzprop, bins=35, title = "Distribuição da Propriedade da Simulação"):
    """
    Plota um histograma da distribuição da propriedade no conjunto de dados de simulação.
    Entrada:
        sim_array_xyzprop: numpy array
                           Dados de simulação com colunas [X, Y, Z, Propriedade].
        bins: int (padrão=35)
              Número de divisões (bins) no histograma.
    Saída:
        Exibe um gráfico de histograma usando matplotlib.
    """
    plt.hist(sim_array_xyzprop[:, -1], bins=bins)
    plt.title(f"{title}")
    plt.xlabel("Valor da Propriedade")
    plt.ylabel("Frequência")
    plt.grid(alpha=0.3)