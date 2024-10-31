import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from pyproj import Geod  # Install using 'pip install pyproj'
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for clean output

# Define the Earth's radius in meters
R_EARTH = 6378137.0

# Function to filter accelerometer data to reduce noise
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    if nyquist <= cutoff:
        raise ValueError("Cutoff frequency must be less than half the sampling rate (Nyquist frequency).")
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Function to interpolate missing data
def interpolate_missing_data(time, data_array):
    mask = np.isfinite(data_array)
    interp_func = interp1d(time[mask], data_array[mask], kind='linear', fill_value='extrapolate')
    return interp_func(time)

# Function to convert GPS coordinates to meters using geodesic calculations
def gps_to_meters(gps_lat, gps_lon, lat0, lon0):
    geod = Geod(ellps='WGS84')
    x = np.zeros(len(gps_lat))
    y = np.zeros(len(gps_lat))
    for i in range(len(gps_lat)):
        az12, az21, dist = geod.inv(lon0, lat0, gps_lon[i], gps_lat[i])
        x[i] = dist * np.cos(np.deg2rad(az12))
        y[i] = dist * np.sin(np.deg2rad(az12))
    return x, y


# Function to synchronize data
def synchronize_data(time_sensor, data_sensor, time_video):
    interp_func = interp1d(time_sensor, data_sensor, kind='linear', fill_value='extrapolate')
    data_video_sync = interp_func(time_video)
    return data_video_sync

# Function to plot the reconstructed path on a map

