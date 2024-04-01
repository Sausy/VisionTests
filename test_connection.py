import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth, color, and infrared streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# Enable color stream
#config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Enable infrared stream of the  camera 
#config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

# Enable IMU
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# Start streaming
profile = pipeline.start(config)

try:
    last_print = time.time()
    while True:
        # Wait for a coherent set of frames: depth, color, gyro, and accel
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        #infrared_frame = frames.first(rs.stream.infrared)

        if not depth_frame or not color_frame:
            print("Could not acquire depth or color frames.")
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #infrared_image = np.asanyarray(infrared_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Resize color image to match the depth image size
        color_image_resized = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))

        # Stack both images horizontally
        images = np.hstack((color_image_resized, depth_colormap))

        # Display images
        #cv2.imshow('Infrared', infrared_image)
        cv2.imshow('RealSense (Color + Depth)', images)
        

        # Print IMU data at a reduced rate for readability
        if accel_frame and gyro_frame and (time.time() - last_print) > 1:
            accel = accel_frame.as_motion_frame().get_motion_data()
            gyro = gyro_frame.as_motion_frame().get_motion_data()
            print(f"Accel: {accel.x:.3f}, {accel.y:.3f}, {accel.z:.3f} | Gyro: {gyro.x:.3f}, {gyro.y:.3f}, {gyro.z:.3f}")
            last_print = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()