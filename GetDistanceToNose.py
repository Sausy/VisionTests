import cv2
import dlib
import numpy as np
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation as R
import math

# Dlib facial landmarks detector
predictor_path = os.path.join("pre_trained_data","shape_predictor_68_face_landmarks.dat")  # Update with the correct path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Configure depth and color streams from RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Enable IMU
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

# Get the intrinsic parameters of the depth sensor
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Helper function to convert from depth to 3D coordinates
def depth_to_xyz(depth_frame, pixel_x, pixel_y, depth_scale):
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    depth = depth_frame.get_distance(pixel_x, pixel_y)
    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth)
    z = np.sqrt(depth**2 - xyz[0]**2 +-xyz[1]**2)
    return xyz[0], xyz[1], z

def to_cartesian(roll, pitch, yaw, length):
    '''
    x = length * math.cos(roll) * math.cos(pitch)
    y = length * math.sin(roll) * math.cos(pitch)
    z = length * math.sin(pitch)
    return x, y, z
    '''
    # Create rotation matrices for roll, pitch, and yaw
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    # Combine the rotations
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    # Initial direction vector (assuming it points along the Z axis)
    direction_vector = np.array([0, 0, 1])

    # Rotate the direction vector
    rotated_vector = np.dot(R, direction_vector)

    # Scale by the given length to get the final Cartesian coordinates
    xyz = length * rotated_vector

    return xyz

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image_dlib = color_image[:, :, ::-1]  # Convert from BGR to RGB

        # Detect faces
        faces = detector(color_image_dlib)
        for face in faces:
            landmarks = predictor(color_image_dlib, face)

            # Get the nose tip position
            nose_tip = landmarks.part(33)  # Note: indexing starts from 0, so the nose tip is at index 33
            x, y = nose_tip.x, nose_tip.y

            # Measure distance
            distance = depth_frame.get_distance(x, y)

            # Convert depth to real-world coordinates
            translation = depth_to_xyz(depth_frame, x, y, depth_scale)
            #translation = [depth_point[0], depth_point[1], depth_point[2]]

            # Retrieve acceleration and gyro data
            accel = accel_frame.as_motion_frame().get_motion_data()
            gyro = gyro_frame.as_motion_frame().get_motion_data()

            # Assuming accel and gyro data are in standard units (m/s^2 and rad/s)
            # Compute roll, pitch, yaw (simplified, assuming static or slow-moving camera)
            roll = math.atan2(accel.y, accel.z)
            pitch = math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2))
            yaw = math.atan2(math.sqrt(accel.x**2 + accel.y**2), accel.z)

            # Convert radians to degrees
            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            

            # Overlay the nose tip and distance on the color image
            cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(color_image, f"Distance: {distance:.2f}m", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.putText(color_image, f"Dist: {distance:.2f}m, X: {depth_point[0]:.2f}m, Y: {depth_point[1]:.2f}m", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # center point
            # Get the resolution from the depth stream profile
            width = depth_profile.width()
            height = depth_profile.height()
            x_center = width // 2  # Center of the image
            y_center = height // 2
            cv2.circle(color_image, (x_center, y_center), 5, (255, 255, 255), -1)  # White circle
            cv2.putText(color_image, f"z: {translation[2]:.2f}m", (x_center-20, y_center+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Draw lines extending from the specified pixel to represent the XYZ axes.
            # Here we simply simulate this in 2D.
            # X axis in red
            cv2.line(color_image, (x_center, y_center), (x, y_center), (0, 0, 255), 2)
            x_text = (x_center + (x - x_center) // 2 ) - 15
            cv2.putText(color_image, f"x: {translation[0]:.2f}m", (x_text, y_center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Y axis in green
            cv2.line(color_image, (x, y), (x, y_center), (0, 255, 0), 2)
            y_text = (y_center + (y - y_center) // 2 ) - 5
            cv2.putText(color_image, f"y: {translation[1]:.2f}m", (x + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Z axis in blue (assuming depth increases in the 'into the screen' direction)
            #cv2.line(color_image, (x, y), (x + 35, y - 35), (255, 0, 0), 2)  # Simulated perspective

            print(f"x: {translation[0]:.5f}, \ty: {translation[1]:.5f}, \tz: {translation[2]:.5f}, \tdistance: {distance:.5f}m, \troll: {roll_deg:.5f}, \tpitch: {pitch_deg:.5f}, \tyaw: {yaw_deg:.5f}")
            translation2 = to_cartesian(roll, pitch, yaw, distance)
            print(f"x: {translation2[0]:.5f}, \ty: {translation2[1]:.5f}, \tz: {translation2[2]:.5f}")

        # Display the image
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()