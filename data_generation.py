
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(path_type, total_time=20, time_step=0.1, noise_std=0.1):
    """
    Generates synthetic sensor data for a specified path type.

    Parameters:
    - path_type: 'circular', 'straight', or 'half_circle_straight'
    - total_time: Total duration of the simulation in seconds
    - time_step: Time interval between data points
    - noise_std: Standard deviation of the measurement noise

    Returns:
    - acc_x: Accelerometer readings along x-axis
    - acc_y: Accelerometer readings along y-axis
    - gyro_z: Gyroscope readings around z-axis
    - mag_x: Magnetometer readings along x-axis
    - mag_y: Magnetometer readings along y-axis
    - gps_lat: GPS latitude readings
    - gps_lon: GPS longitude readings
    - time_array: Array of time stamps
    """
    # Time array
    time_array = np.arange(0, total_time, time_step)
    num_samples = len(time_array)

    # Initialize arrays
    x_true = np.zeros(num_samples)
    y_true = np.zeros(num_samples)
    vx_true = np.zeros(num_samples)
    vy_true = np.zeros(num_samples)
    theta_true = np.zeros(num_samples)

    acc_x = np.zeros(num_samples)
    acc_y = np.zeros(num_samples)
    gyro_z = np.zeros(num_samples)
    mag_x = np.zeros(num_samples)
    mag_y = np.zeros(num_samples)
    gps_lat = np.zeros(num_samples)
    gps_lon = np.zeros(num_samples)

    # Simulation parameters
    dt = time_step
    speed = 1.0  # meters per second
    radius = 10.0  # meters
    angular_speed = speed / radius  # radians per second

    # Earth's radius in meters (for GPS simulation)
    R_earth = 6378137.0
    lat0 = 0.0  # Reference latitude in degrees
    lon0 = 0.0  # Reference longitude in degrees

    for k in range(num_samples):
        t = time_array[k]

        if path_type == 'circular':
            # Circular motion
            theta = angular_speed * t
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vx = -radius * angular_speed * np.sin(theta)
            vy = radius * angular_speed * np.cos(theta)
            ax = -radius * angular_speed**2 * np.cos(theta)
            ay = -radius * angular_speed**2 * np.sin(theta)
            theta_heading = theta + np.pi / 2  # Heading tangent to the circle
            omega_z = angular_speed  # Constant angular velocity
        elif path_type == 'straight':
            # Straight line motion along x-axis
            x = speed * t
            y = 0.0
            vx = speed
            vy = 0.0
            ax = 0.0
            ay = 0.0
            theta_heading = 0.0  # Facing along x-axis
            omega_z = 0.0
        elif path_type == 'half_circle_straight':
            # Half circle followed by straight line
            if t <= total_time / 2:
                # First half: circular motion
                theta = angular_speed * t
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                vx = -radius * angular_speed * np.sin(theta)
                vy = radius * angular_speed * np.cos(theta)
                ax = -radius * angular_speed**2 * np.cos(theta)
                ay = -radius * angular_speed**2 * np.sin(theta)
                theta_heading = theta + np.pi / 2
                omega_z = angular_speed
            else:
                # Second half: straight line motion along x-axis
                t_straight = t - total_time / 2
                x0 = radius * np.cos(angular_speed * total_time / 2)
                x = x0 + speed * t_straight
                y = radius * np.sin(angular_speed * total_time / 2)
                vx = speed
                vy = 0.0
                ax = 0.0
                ay = 0.0
                theta_heading = 0.0
                omega_z = 0.0
        else:
            raise ValueError("Invalid path_type. Choose 'circular', 'straight', or 'half_circle_straight'.")

        # Store true values
        x_true[k] = x
        y_true[k] = y
        vx_true[k] = vx
        vy_true[k] = vy
        theta_true[k] = theta_heading

        # Simulate sensor readings with noise
        acc_x[k] = ax + np.random.normal(0, noise_std)
        acc_y[k] = ay + np.random.normal(0, noise_std)
        gyro_z[k] = omega_z + np.random.normal(0, noise_std * 0.1)

        # Magnetometer readings (simulate Earth's magnetic field along x-axis)
        mag_x_global = np.cos(theta_heading)
        mag_y_global = np.sin(theta_heading)
        mag_x[k] = mag_x_global + np.random.normal(0, noise_std * 0.05)
        mag_y[k] = mag_y_global + np.random.normal(0, noise_std * 0.05)

        # Simulate GPS readings with noise
        # Convert x, y to latitude and longitude for simulation
        delta_lat = y / R_earth * (180 / np.pi)
        delta_lon = x / (R_earth * np.cos(np.pi * lat0 / 180)) * (180 / np.pi)
        gps_lat[k] = lat0 + delta_lat + np.random.normal(0, noise_std * 1e-5)
        gps_lon[k] = lon0 + delta_lon + np.random.normal(0, noise_std * 1e-5)

    # Plot the true path for visualization
    plt.figure(figsize=(8, 6))
    plt.plot(x_true, y_true, label='True Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'True Path: {path_type.capitalize()}')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()

    return acc_x, acc_y, gyro_z, mag_x, mag_y, gps_lat, gps_lon, time_array
