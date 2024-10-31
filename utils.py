
import numpy as np
import pandas as pd
import cv2
from preprocessing import interpolate_missing_data,butter_lowpass_filter,synchronize_data
from kalman_filter import full_extended_kalman_filter
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

    # Check for required columns
    required_columns = {'time', 'acc_x', 'acc_y', 'acc_z', 'gyro_z', 'gps_lat', 'gps_lon'}
    if not required_columns.issubset(data.columns):
        print("Error: Sensor data file is missing required columns.")
        return

    # Extract sensor data columns
    time_sensor = data['time'].values  # Timestamps (in seconds)
    acc_x = data['acc_x'].values
    acc_y = data['acc_y'].values
    acc_z = data['acc_z'].values
    gyro_z = data['gyro_z'].values  # Assuming gyro_z is the angular rate around the Z-axis
    gps_lat = data['gps_lat'].values
    gps_lon = data['gps_lon'].values

    # Handle missing data by interpolation
    acc_x = interpolate_missing_data(time_sensor, acc_x)
    acc_y = interpolate_missing_data(time_sensor, acc_y)
    acc_z = interpolate_missing_data(time_sensor, acc_z)
    gyro_z = interpolate_missing_data(time_sensor, gyro_z)
    gps_lat = interpolate_missing_data(time_sensor, gps_lat)
    gps_lon = interpolate_missing_data(time_sensor, gps_lon)

    # Filter accelerometer data
    fs = 50  # Sampling frequency in Hz (should be obtained from data or metadata)
    cutoff = 5  # Increased cutoff frequency to better capture walking dynamics
    try:
        acc_x_filtered = butter_lowpass_filter(acc_x, cutoff, fs)
        acc_y_filtered = butter_lowpass_filter(acc_y, cutoff, fs)
        acc_z_filtered = butter_lowpass_filter(acc_z, cutoff, fs)
    except ValueError as e:
        print(f"Filtering Error: {e}")
        return

    # Correct for sensor bias (assuming stationary at the start)
    acc_x_filtered -= np.mean(acc_x_filtered[:fs])  # Subtract mean of the first second
    acc_y_filtered -= np.mean(acc_y_filtered[:fs])
    acc_z_filtered -= np.mean(acc_z_filtered[:fs])
    gyro_z -= np.mean(gyro_z[:fs])  # Correct gyroscope bias

    # Apply the full Extended Kalman Filter
    x_estimated, y_estimated, theta_estimated = full_extended_kalman_filter(
        acc_x_filtered, acc_y_filtered, gyro_z, gps_lat, gps_lon, time_sensor
    )

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
