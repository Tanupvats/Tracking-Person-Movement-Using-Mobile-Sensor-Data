import numpy as np
import pandas as pd
import cv2
from preprocessing import interpolate_missing_data, butter_lowpass_filter, synchronize_data
from kalman_filter import ekf_accelerometer_gps
from kalman_filter import full_extended_kalman_filter, ekf_full_sensors 
from visualization import plot_path_on_map

def main():
    # Assuming sensor data is collected as CSV and video frames from a video file
    sensor_data_file = "sensor_data.csv"  # CSV file containing accelerometer, gyroscope, GPS data
    video_file = "car_video.mp4"

    # Load sensor data with error handling
    try:
        data = pd.read_csv(sensor_data_file)
    except FileNotFoundError:
        print(f"Error: Sensor data file '{sensor_data_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: Sensor data file is empty.")
        return

    # Check for available sensor data columns
    available_columns = set(data.columns)
    required_columns = {'time', 'acc_x', 'acc_y', 'acc_z'}
    optional_columns = {'gyro_z', 'gps_lat', 'gps_lon', 'mag_heading'}

    if not required_columns.issubset(available_columns):
        print("Error: Sensor data file is missing required accelerometer columns.")
        return

    # Extract required sensor data columns
    time_sensor = data['time'].values  # Timestamps (in seconds)
    acc_x = data['acc_x'].values
    acc_y = data['acc_y'].values
    acc_z = data['acc_z'].values

    # Handle missing data by interpolation
    acc_x = interpolate_missing_data(time_sensor, acc_x)
    acc_y = interpolate_missing_data(time_sensor, acc_y)
    acc_z = interpolate_missing_data(time_sensor, acc_z)

    # Filter accelerometer data
    fs = 50  # Sampling frequency in Hz (should be obtained from data or metadata)
    cutoff = 5  # Increased cutoff frequency for better filtering
    acc_x_filtered = butter_lowpass_filter(acc_x, cutoff, fs)
    acc_y_filtered = butter_lowpass_filter(acc_y, cutoff, fs)
    acc_z_filtered = butter_lowpass_filter(acc_z, cutoff, fs)

    # Determine which Kalman filter to apply based on available columns
    if {'gps_lat', 'gps_lon'}.issubset(available_columns):
        gps_lat = interpolate_missing_data(time_sensor, data['gps_lat'].values)
        gps_lon = interpolate_missing_data(time_sensor, data['gps_lon'].values)

        if 'gyro_z' in available_columns:
            gyro_z = interpolate_missing_data(time_sensor, data['gyro_z'].values)
            # If both GPS and gyroscope data are available, use EKF with accelerometer, GPS, and gyroscope
            x_estimated, y_estimated, theta_estimated = full_extended_kalman_filter(
                acc_x_filtered, acc_y_filtered, gyro_z, gps_lat, gps_lon, time_sensor
            )
        elif 'mag_heading' in available_columns:
            mag_heading = interpolate_missing_data(time_sensor, data['mag_heading'].values)
            # If GPS and magnetometer data are available, use EKF with accelerometer, GPS, and magnetometer
            x_estimated, y_estimated, theta_estimated = ekf_full_sensors(
                acc_x_filtered, acc_y_filtered, gps_lat, gps_lon, time_sensor, mag_heading
            )
        else:
            # Use EKF with accelerometer and GPS data
            x_estimated, y_estimated, theta_estimated = ekf_accelerometer_gps(
                acc_x_filtered, acc_y_filtered, gps_lat, gps_lon, time_sensor
            )
    else:
        print("Error: GPS data is not available for EKF.")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_video = total_frames / fps
    time_video = np.linspace(0, duration_video, total_frames)

    # Synchronize estimated positions with video frames
    position_x_video = synchronize_data(time_sensor, x_estimated, time_video)
    position_y_video = synchronize_data(time_sensor, y_estimated, time_video)
    theta_video = synchronize_data(time_sensor, theta_estimated, time_video)

    # Process each frame of the video
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the corresponding synchronized data
        current_position = (position_x_video[frame_idx], position_y_video[frame_idx])
        current_theta = theta_video[frame_idx]

        # Overlay position and orientation information
        text = f"Position: X={current_position[0]:.2f}m, Y={current_position[1]:.2f}m"
        theta_text = f"Orientation: θ={np.rad2deg(current_theta):.2f}°"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, theta_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Tracking Video', frame)
        frame_idx += 1

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Plot the reconstructed path on a map
    plot_path_on_map(x_estimated, y_estimated)

if __name__ == "__main__":
    main()
