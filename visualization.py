
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import folium

def plot_2d_path(x_true, y_true, x_estimated, y_estimated, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    plt.plot(x_true, y_true, label='True Path', linewidth=2, color='blue')
    plt.plot(x_estimated, y_estimated, label='EKF Estimated Path', linestyle='--', linewidth=2, color='green')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_3d_path(x_true, y_true, z_true, x_estimated, y_estimated, z_estimated, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_true, y_true, z_true, label='True Path', linewidth=2, color='blue')
    ax.plot(x_estimated, y_estimated, z_estimated, label='EKF Estimated Path', linestyle='--', linewidth=2, color='green')
    ax.set_title(title)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.legend()
    plt.show()

def plot_path_on_map(x_estimated, y_estimated):
    try:
        
        # Convert estimated positions to GPS coordinates
        geod = Geod(ellps='WGS84')
        latitudes = []
        longitudes = []
        lat0, lon0 = 0.0, 0.0  # Origin (will be set later)

        for i in range(len(x_estimated)):
            if i == 0:
                # Initial position
                lat0, lon0 = 0.0, 0.0  # You can set this to the initial GPS coordinates
                latitudes.append(lat0)
                longitudes.append(lon0)
            else:
                azimuth = np.rad2deg(np.arctan2(y_estimated[i] - y_estimated[i - 1], x_estimated[i] - x_estimated[i - 1]))
                distance = np.hypot(x_estimated[i] - x_estimated[i - 1], y_estimated[i] - y_estimated[i - 1])
                lon, lat, _ = geod.fwd(longitudes[-1], latitudes[-1], azimuth, distance)
                latitudes.append(lat)
                longitudes.append(lon)

        # Create a map centered at the average location
        avg_lat = np.mean(latitudes)
        avg_lon = np.mean(longitudes)
        map_ = folium.Map(location=[avg_lat, avg_lon], zoom_start=20)

        # Add the path to the map
        path = list(zip(latitudes, longitudes))
        folium.PolyLine(path, color='blue', weight=2.5, opacity=1).add_to(map_)

        # Display the map
        map_.save('path_map.html')
        print("Map has been saved as 'path_map.html'. Open it in a web browser to view.")
    except ImportError:
        print("Folium library not installed. Skipping map plot.")
        print("Install it using 'pip install folium' to enable map plotting.")