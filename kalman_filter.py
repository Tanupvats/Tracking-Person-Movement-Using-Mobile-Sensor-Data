
import numpy as np
from ekf import ExtendedKalmanFilter
from preprocessing import gps_to_meters

def ekf_accelerometer_gps(acc_x, acc_y, gps_lat, gps_lon, time):
    num_samples = len(time)

    # Compute time differences
    dt = np.diff(time)
    dt = np.insert(dt, 0, dt[0])  # Assume first dt is same as second
    dt[dt <= 0] = 1e-5  # Replace zero or negative dt with a small positive value

    # Initial state vector [x, y, vx, vy, bax, bay]
    state = np.zeros(6)
    # Initial covariance matrix with high uncertainty
    P = np.eye(6) * 500

    # Process noise covariance (tuned based on expected sensor noise characteristics)
    Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.001, 0.001])

    # Measurement noise covariance for GPS
    R_gps = np.diag([5.0, 5.0])  # GPS position measurement noise

    # Store estimates
    state_estimates = np.zeros((num_samples, 6))

    # Initial GPS position in meters
    lat0, lon0 = gps_lat[0], gps_lon[0]
    x_gps, y_gps = gps_to_meters(gps_lat, gps_lon, lat0, lon0)

    for k in range(num_samples):
        # Time step
        dt_k = dt[k]

        # **Prediction Step**

        # Extract previous state
        x_prev, y_prev, vx_prev, vy_prev, bax_prev, bay_prev = state

        # Control inputs (accelerations corrected for biases)
        acc_x_corrected = acc_x[k] - bax_prev
        acc_y_corrected = acc_y[k] - bay_prev

        # Predict next state
        x_pred = x_prev + vx_prev * dt_k + 0.5 * acc_x_corrected * dt_k ** 2
        y_pred = y_prev + vy_prev * dt_k + 0.5 * acc_y_corrected * dt_k ** 2
        vx_pred = vx_prev + acc_x_corrected * dt_k
        vy_pred = vy_prev + acc_y_corrected * dt_k
        bax_pred = bax_prev  # Biases are assumed to change slowly
        bay_pred = bay_prev

        # Predicted state
        state_pred = np.array([x_pred, y_pred, vx_pred, vy_pred, bax_pred, bay_pred])

        # Jacobian of the state transition function (F_k)
        F_k = np.eye(6)
        F_k[0, 2] = dt_k
        F_k[1, 3] = dt_k
        F_k[0, 4] = -0.5 * dt_k ** 2
        F_k[1, 5] = -0.5 * dt_k ** 2
        F_k[2, 4] = -dt_k
        F_k[3, 5] = -dt_k

        # Predict the covariance
        P_pred = F_k @ P @ F_k.T + Q

        # **Update Step**

        # Measurement vector
        z_gps = np.array([x_gps[k], y_gps[k]])

        # Measurement function for GPS
        H_gps = np.zeros((2, 6))
        H_gps[0, 0] = 1  # Partial derivative of position x w.r.t state x
        H_gps[1, 1] = 1  # Partial derivative of position y w.r.t state y

        # Innovation (measurement residual)
        y_gps_meas = z_gps - state_pred[:2]

        # Innovation covariance
        S_gps = H_gps @ P_pred @ H_gps.T + R_gps

        # Kalman Gain
        K_gps = P_pred @ H_gps.T @ np.linalg.inv(S_gps)

        # Update state estimate with GPS measurement
        state_upd = state_pred + K_gps @ y_gps_meas

        # Update covariance estimate
        P_upd = (np.eye(6) - K_gps @ H_gps) @ P_pred

        # **Store the updated state**
        state = state_upd
        P = P_upd
        state_estimates[k] = state

    # Extract positions
    x_estimated = state_estimates[:, 0]
    y_estimated = state_estimates[:, 1]

    return x_estimated, y_estimated


def full_extended_kalman_filter(acc_x, acc_y, gyro_z, gps_lat, gps_lon, time):
    num_samples = len(time)

    # Compute time differences
    dt = np.diff(time)
    dt = np.insert(dt, 0, dt[0])  # Assume first dt is the same as the second
    dt[dt <= 0] = 1e-5  # Replace zero or negative dt with a small positive value

    # Initial state vector [x, y, vx, vy, theta, bax, bay, bgz]
    state = np.zeros(8)
    # Initial covariance matrix with high uncertainty
    P = np.eye(8) * 500

    # Process noise covariance (tuned based on expected sensor noise characteristics)
    Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.05, 0.001, 0.001, 0.0001])

    # Measurement noise covariance for GPS
    R_gps = np.diag([5.0, 5.0])  # GPS position measurement noise

    # Store estimates
    state_estimates = np.zeros((num_samples, 8))

    # Initial GPS position in meters
    lat0, lon0 = gps_lat[0], gps_lon[0]
    x_gps, y_gps = gps_to_meters(gps_lat, gps_lon, lat0, lon0)

    for k in range(num_samples):
        # Time step
        dt_k = dt[k]

        # **Prediction Step**

        # Extract previous state
        x_prev, y_prev, vx_prev, vy_prev, theta_prev, bax_prev, bay_prev, bgz_prev = state

        # Control inputs (accelerations corrected for biases)
        acc_x_corrected = acc_x[k] - bax_prev
        acc_y_corrected = acc_y[k] - bay_prev

        # Rotate accelerations to global frame
        acc_global_x = acc_x_corrected * np.cos(theta_prev) - acc_y_corrected * np.sin(theta_prev)
        acc_global_y = acc_x_corrected * np.sin(theta_prev) + acc_y_corrected * np.cos(theta_prev)

        # Predict next state
        x_pred = x_prev + vx_prev * dt_k + 0.5 * acc_global_x * dt_k ** 2
        y_pred = y_prev + vy_prev * dt_k + 0.5 * acc_global_y * dt_k ** 2
        vx_pred = vx_prev + acc_global_x * dt_k
        vy_pred = vy_prev + acc_global_y * dt_k
        theta_pred = theta_prev + (gyro_z[k] - bgz_prev) * dt_k
        bax_pred = bax_prev  # Biases are assumed to change slowly
        bay_pred = bay_prev
        bgz_pred = bgz_prev

        # Normalize theta to [-pi, pi]
        theta_pred = (theta_pred + np.pi) % (2 * np.pi) - np.pi

        # Predicted state
        state_pred = np.array([x_pred, y_pred, vx_pred, vy_pred, theta_pred, bax_pred, bay_pred, bgz_pred])

        # Jacobian of the state transition function (F_k)
        F_k = np.eye(8)
        F_k[0, 2] = dt_k
        F_k[1, 3] = dt_k
        F_k[0, 4] = -0.5 * (acc_x_corrected * np.sin(theta_prev) + acc_y_corrected * np.cos(theta_prev)) * dt_k ** 2
        F_k[1, 4] = 0.5 * (acc_x_corrected * np.cos(theta_prev) - acc_y_corrected * np.sin(theta_prev)) * dt_k ** 2
        F_k[2, 4] = -(acc_x_corrected * np.sin(theta_prev) + acc_y_corrected * np.cos(theta_prev)) * dt_k
        F_k[3, 4] = (acc_x_corrected * np.cos(theta_prev) - acc_y_corrected * np.sin(theta_prev)) * dt_k
        F_k[0, 5] = -0.5 * np.cos(theta_prev) * dt_k ** 2
        F_k[0, 6] = 0.5 * np.sin(theta_prev) * dt_k ** 2
        F_k[1, 5] = -0.5 * np.sin(theta_prev) * dt_k ** 2
        F_k[1, 6] = -0.5 * np.cos(theta_prev) * dt_k ** 2
        F_k[2, 5] = -np.cos(theta_prev) * dt_k
        F_k[2, 6] = np.sin(theta_prev) * dt_k
        F_k[3, 5] = -np.sin(theta_prev) * dt_k
        F_k[3, 6] = -np.cos(theta_prev) * dt_k
        F_k[4, 7] = -dt_k

        # Predict the covariance
        P_pred = F_k @ P @ F_k.T + Q

        # **Update Step**

        # Measurement vector
        z_gps = np.array([x_gps[k], y_gps[k]])

        # Measurement function for GPS
        H_gps = np.zeros((2, 8))
        H_gps[0, 0] = 1  # Partial derivative of position x w.r.t state x
        H_gps[1, 1] = 1  # Partial derivative of position y w.r.t state y

        # Innovation (measurement residual)
        y_gps_meas = z_gps - np.array([x_pred, y_pred])

        # Innovation covariance
        S_gps = H_gps @ P_pred @ H_gps.T + R_gps

        # Kalman Gain
        K_gps = P_pred @ H_gps.T @ np.linalg.inv(S_gps)

        # Update state estimate with GPS measurement
        state_upd = state_pred + K_gps @ y_gps_meas

        # Update covariance estimate
        P_upd = (np.eye(8) - K_gps @ H_gps) @ P_pred

        # Normalize theta to [-pi, pi] after update
        state_upd[4] = (state_upd[4] + np.pi) % (2 * np.pi) - np.pi

        # **Store the updated state**
        state = state_upd
        P = P_upd
        state_estimates[k] = state

    # Extract positions and orientations
    x_estimated = state_estimates[:, 0]
    y_estimated = state_estimates[:, 1]
    theta_estimated = state_estimates[:, 4]

    # Normalize all theta_estimated to [-pi, pi]
    theta_estimated = (theta_estimated + np.pi) % (2 * np.pi) - np.pi

    return x_estimated, y_estimated, theta_estimated


def ekf_full_sensors(acc_x, acc_y, gyro_z, mag_x, mag_y, gps_lat, gps_lon, time):
    num_samples = len(time)

    # Compute time differences
    dt = np.diff(time)
    dt = np.insert(dt, 0, dt[0])  # Assume first dt is same as second
    dt[dt <= 0] = 1e-5  # Replace zero or negative dt with a small positive value

    # Initial state vector [x, y, vx, vy, theta, bax, bay, bgz]
    state = np.zeros(8)
    # Initial covariance matrix with high uncertainty
    P = np.eye(8) * 500

    # Process noise covariance (tuned based on expected sensor noise characteristics)
    Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.05, 0.001, 0.001, 0.0001])

    # Measurement noise covariance
    R_gps = np.diag([5.0, 5.0])  # GPS position measurement noise
    R_mag = np.array([[0.05]])    # Magnetometer measurement noise

    # Store estimates
    state_estimates = np.zeros((num_samples, 8))

    # Initial GPS position in meters
    lat0, lon0 = gps_lat[0], gps_lon[0]
    x_gps, y_gps = gps_to_meters(gps_lat, gps_lon, lat0, lon0)

    for k in range(num_samples):
        # Time step
        dt_k = dt[k]

        # **Prediction Step**

        # Extract previous state
        x_prev, y_prev, vx_prev, vy_prev, theta_prev, bax_prev, bay_prev, bgz_prev = state

        # Control inputs (accelerations corrected for biases and rotated)
        acc_x_corrected = acc_x[k] - bax_prev
        acc_y_corrected = acc_y[k] - bay_prev

        # Rotate accelerations to global frame
        acc_global_x = acc_x_corrected * np.cos(theta_prev) - acc_y_corrected * np.sin(theta_prev)
        acc_global_y = acc_x_corrected * np.sin(theta_prev) + acc_y_corrected * np.cos(theta_prev)

        # Predict next state
        x_pred = x_prev + vx_prev * dt_k + 0.5 * acc_global_x * dt_k ** 2
        y_pred = y_prev + vy_prev * dt_k + 0.5 * acc_global_y * dt_k ** 2
        vx_pred = vx_prev + acc_global_x * dt_k
        vy_pred = vy_prev + acc_global_y * dt_k
        theta_pred = theta_prev + (gyro_z[k] - bgz_prev) * dt_k
        bax_pred = bax_prev  # Biases are assumed to change slowly
        bay_pred = bay_prev
        bgz_pred = bgz_prev

        # Normalize theta to [-pi, pi]
        theta_pred = (theta_pred + np.pi) % (2 * np.pi) - np.pi

        # Predicted state
        state_pred = np.array([x_pred, y_pred, vx_pred, vy_pred, theta_pred, bax_pred, bay_pred, bgz_pred])

        # Jacobian of the state transition function (F_k)
        F_k = np.eye(8)
        F_k[0, 2] = dt_k
        F_k[1, 3] = dt_k
        F_k[0, 4] = -0.5 * (acc_x_corrected * np.sin(theta_prev) + acc_y_corrected * np.cos(theta_prev)) * dt_k ** 2
        F_k[1, 4] = 0.5 * (acc_x_corrected * np.cos(theta_prev) - acc_y_corrected * np.sin(theta_prev)) * dt_k ** 2
        F_k[2, 4] = -(acc_x_corrected * np.sin(theta_prev) + acc_y_corrected * np.cos(theta_prev)) * dt_k
        F_k[3, 4] = (acc_x_corrected * np.cos(theta_prev) - acc_y_corrected * np.sin(theta_prev)) * dt_k
        F_k[0, 5] = -0.5 * np.cos(theta_prev) * dt_k ** 2
        F_k[0, 6] = 0.5 * np.sin(theta_prev) * dt_k ** 2
        F_k[1, 5] = -0.5 * np.sin(theta_prev) * dt_k ** 2
        F_k[1, 6] = -0.5 * np.cos(theta_prev) * dt_k ** 2
        F_k[2, 5] = -np.cos(theta_prev) * dt_k
        F_k[2, 6] = np.sin(theta_prev) * dt_k
        F_k[3, 5] = -np.sin(theta_prev) * dt_k
        F_k[3, 6] = -np.cos(theta_prev) * dt_k
        F_k[4, 7] = -dt_k

        # Predict the covariance
        P_pred = F_k @ P @ F_k.T + Q

        # **Update Step**

        # Measurement vector (GPS and magnetometer)
        z_gps = np.array([x_gps[k], y_gps[k]])

        # Magnetometer provides heading measurement
        mag_heading = np.arctan2(mag_y[k], mag_x[k])
        mag_heading = (mag_heading + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Measurement function for GPS
        H_gps = np.zeros((2, 8))
        H_gps[0, 0] = 1  # Partial derivative of position x w.r.t state x
        H_gps[1, 1] = 1  # Partial derivative of position y w.r.t state y

        # Innovation (measurement residual) for GPS
        y_gps_meas = z_gps - state_pred[:2]

        # Innovation covariance for GPS
        S_gps = H_gps @ P_pred @ H_gps.T + R_gps

        # Kalman Gain for GPS
        K_gps = P_pred @ H_gps.T @ np.linalg.inv(S_gps)

        # Update state estimate with GPS measurement
        state_upd = state_pred + K_gps @ y_gps_meas

        # Update covariance estimate
        P_upd = (np.eye(8) - K_gps @ H_gps) @ P_pred

        # **Second Update with Magnetometer**

        # Measurement function for magnetometer
        H_mag = np.zeros((1, 8))
        H_mag[0, 4] = 1  # Partial derivative of theta w.r.t theta

        # Innovation (measurement residual) for magnetometer
        y_mag_meas = mag_heading - state_upd[4]
        # Normalize the angle difference to [-pi, pi]
        y_mag_meas = (y_mag_meas + np.pi) % (2 * np.pi) - np.pi

        # Innovation covariance for magnetometer
        S_mag = H_mag @ P_upd @ H_mag.T + R_mag

        # Kalman Gain for magnetometer
        K_mag = P_upd @ H_mag.T @ np.linalg.inv(S_mag)

        # Update state estimate with magnetometer measurement
        state_upd = state_upd + K_mag @ y_mag_meas

        # Update covariance estimate
        P_upd = (np.eye(8) - K_mag @ H_mag) @ P_upd

        # Normalize theta to [-pi, pi] after update
        state_upd[4] = (state_upd[4] + np.pi) % (2 * np.pi) - np.pi

        # **Store the updated state**
        state = state_upd
        P = P_upd
        state_estimates[k] = state

    # Extract positions and orientations
    x_estimated = state_estimates[:, 0]
    y_estimated = state_estimates[:, 1]
    theta_estimated = state_estimates[:, 4]

    # Normalize all theta_estimated to [-pi, pi]
    theta_estimated = (theta_estimated + np.pi) % (2 * np.pi) - np.pi

    return x_estimated, y_estimated, theta_estimated
