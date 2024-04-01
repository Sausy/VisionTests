from openni import openni2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
import threading
from queue import Queue
import time

class ColorAdjustmentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Color Adjustment and Object Detection App")
        self.color_correction_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.create_widgets()
        # Queue for communication between threads
        self.queue = Queue()  
        # Queue for GUI elements to be communicated between threads
        self.gui_queue = Queue()

    def create_widgets(self):
        # Create GUI widgets
        self.create_sliders()
        self.initialize_openni()
        self.create_checkbox()
        self.create_display_button()

    def create_sliders(self):
        # Create sliders for color adjustment matrix
        for i in range(3):
            for j in range(3):
                slider_label = ttk.Label(self.master, text=f"M[{i}][{j}]")
                slider_label.grid(row=i, column=j, padx=5, pady=5)

                slider_var = tk.DoubleVar()
                slider_var.set(self.color_correction_matrix[i, j])
                slider = ttk.Scale(self.master, from_=0.0, to=2.0, length=200, variable=slider_var, orient=tk.HORIZONTAL)
                slider.grid(row=i, column=j + 1, padx=5, pady=5)

                slider_var.trace_add("write", self.update_matrix)

    def initialize_openni(self):
        # Initialize OpenNI and create color/depth streams
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        self.color_stream = self.dev.create_color_stream()
        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream.start()
        self.depth_stream.start()

    def create_checkbox(self):
        # Create checkbox for object detection
        self.detect_cocacola_var = tk.BooleanVar()
        self.detect_cocacola_var.set(False)
        self.detect_cocacola_checkbox = ttk.Checkbutton(self.master, text="Detect Coca-Cola Bottles", variable=self.detect_cocacola_var)
        self.detect_cocacola_checkbox.grid(row=3, column=0, columnspan=4, pady=10)

    def create_display_button(self):
        # Create button to display frames
        self.display_button = ttk.Button(self.master, text="Display Frames", command=self.display_frames)
        self.display_button.grid(row=4, column=0, columnspan=4, pady=10)

    def update_matrix(self, *args):
        # Update color correction matrix based on slider values
        for i in range(3):
            for j in range(3):
                self.color_correction_matrix[i, j] = self.sliders[i * 3 + j].get()

    def detect_cocacola_thread(self):
        darknet_dir = "./darknet"  # Specify the path to your darknet directory

        # Load YOLOv3 model and configuration
        yolo_weights = os.path.join(darknet_dir, "yolov3.weights")
        yolo_cfg = os.path.join(darknet_dir, "yolov3.cfg")
        coco_names = os.path.join(darknet_dir, "coco.names")

        net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        with open(coco_names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        #clear the queue before starting the loop
        self.queue = Queue()

        while True:
            # Get frames from the main thread
            color_image_corrected, depth_frame = self.queue.get()
            print("got frame")

            #if self.detect_cocacola_var.get() and self.detect_cocacola_bottle(color_image_corrected, net, classes):
            #    self.add_text_overlay(color_image_corrected, "Coca-Cola Bottle Detected!")
            if self.detect_cocacola_bottle(color_image_corrected, net, classes, depth_frame):
                print("Coca-Cola bottle detected")
                self.gui_queue.put((color_image_corrected, "Coca-Cola Bottle Detected!"))
            #    self.add_text_overlay(color_image_corrected, "Coca-Cola Bottle Detected!")
            else:
                self.gui_queue.put((color_image_corrected, ""))

            #clear the queue
            self.queue = Queue()
    
    def convert_depth_to_distance(self, depth_value):
        # Implement the conversion from depth value to distance based on your camera's specifications
        # This is a placeholder; you need to replace it with the actual conversion formula for your camera
        # Consult your camera's documentation for information on depth-to-distance conversion
        # Here's a simple example assuming linear conversion (replace this with the actual formula):
        # distance = 1.0 / depth_value

        # Replace the above line with your actual conversion formula
        distance = depth_value

        return distance

    def convert_coordinates_to_original(self, x, y, w, h, original_image_shape):
        # Convert bounding box coordinates from resized image to original image size
        x = int(x * original_image_shape[1] / 416)
        y = int(y * original_image_shape[0] / 416)
        w = int(w * original_image_shape[1] / 416)
        h = int(h * original_image_shape[0] / 416)
        return x, y, w, h

    def detect_cocacola_bottle(self, frame, net, classes, depth_frame):
        resized_image = frame
        # Convert the frame to a blob to be used as input to the neural network
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Get the output layer names
        out_layer_names = net.getUnconnectedOutLayersNames()

        # Run forward pass and get predictions
        outs = net.forward(out_layer_names)

        # Iterate through the predictions
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if the detected object is a bottle and has high confidence
                if confidence > 0.5 and classes[class_id] == "bottle":
                    # Get the coordinates of the bounding box
                    box = detection[0:4] * np.array([resized_image.shape[1], resized_image.shape[0], resized_image.shape[1], resized_image.shape[0]])
                    (x, y, w, h) = box.astype("int")

                    # Convert the bounding box coordinates to the original image size
                    x, y, w, h = self.convert_coordinates_to_original(x, y, w, h, frame.shape)

                    # Print the position of the Coca-Cola bottle
                    print(f"Coca-Cola bottle detected at position (x={x}, y={y}), width={w}, height={h}")

                    # Calculate the distance using the depth information
                    depth_value = np.mean(depth_frame[y:y+h, x:x+w])  # Use the mean depth value within the bounding box
                    distance = self.convert_depth_to_distance(depth_value)

                    # Print the distance
                    print(f"Coca-Cola bottle detected at a distance of {distance:.2f} meters.")
                    print("Depth value:", depth_frame[int(y), int(x)])
                    print("Depth value:", depth_frame[int(x), int(y)])
                    print("Depth value center:", depth_frame[int(y+h/2), int(x+w/2)])
                    print("Depth value center:", depth_frame[int(x+20), int(y+20)])

                    # TODO: Add additional check for Coca-Cola brand if needed
                    return True  # Coca-Cola bottle is detected

        return False  # Coca-Cola bottle is not detected

    def start_threads(self):
        # Start the object detection thread
        cocacola_thread = threading.Thread(target=self.detect_cocacola_thread, daemon=True)
        cocacola_thread.start()

    def display_frames(self):
        darknet_dir = "./darknet"  # Specify the path to your darknet directory
        message = "" 

        try:
            while True:
                color_frame, depth_frame = self.read_frames()

                color_image_corrected = self.apply_color_correction(color_frame)

                # Send frames to the object detection thread
                self.queue.put((color_image_corrected, depth_frame))

                

                try:
                    # Get text overlay from the object detection thread
                    foo_image, message = self.gui_queue.get_nowait()                    
                except:
                    #message = ""
                    pass
                
                # Update the GUI based on the received message
                self.add_text_overlay(color_image_corrected, message)

                # Display frames
                depth_colormap = self.process_depth_frame(depth_frame)
                self.show_frames(color_image_corrected, depth_colormap)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.stop_openni()
            cv2.destroyAllWindows()

    def read_frames(self):
        # Read color and depth frames from OpenNI streams
        color_frame = self.color_stream.read_frame()
        depth_frame = self.depth_stream.read_frame()
        color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
        color_image = color_data.reshape((color_frame.height, color_frame.width, 3))
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
        depth_image = depth_data.reshape((depth_frame.height, depth_frame.width))
        return color_image, depth_image

    def apply_color_correction(self, color_image):
        # Apply color correction matrix to the color frame
        color_image_corrected = cv2.transform(color_image, self.color_correction_matrix)
        return color_image_corrected

    def add_text_overlay(self, frame, text):
        # Add text overlay to the frame
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def process_depth_frame(self, depth_frame):
        # Process depth frame and create colormap
        normalized_depth = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
        return depth_colormap

    def show_frames(self, color_frame, depth_colormap):
        # Display color and depth frames
        cv2.imshow('Color Frame', color_frame)
        cv2.imshow('Depth Frame', depth_colormap)

    def stop_openni(self):
        # Stop OpenNI streams and unload
        self.color_stream.stop()
        self.depth_stream.stop()
        openni2.unload()

def check_cuda_support():
    # Check if OpenCV is compiled with CUDA support
    opencv_cuda_support = cv2.cuda.getCudaEnabledDeviceCount() > 0

    # Check if YOLO is compiled with CUDA support
    try:
        yolo_info = cv2.dnn.getAvailableBackends()
        yolo_cuda_support = cv2.dnn.DNN_BACKEND_CUDA in yolo_info
    except AttributeError:
        # Older versions of OpenCV may not have getAvailableBackends
        yolo_cuda_support = False

    return opencv_cuda_support, yolo_cuda_support


def main():
    # Check CUDA support
    opencv_cuda, yolo_cuda = check_cuda_support()

    # Print the results
    print(f"OpenCV CUDA Support: {opencv_cuda}")
    print(f"YOLO CUDA Support: {yolo_cuda}")

    root = tk.Tk()
    app = ColorAdjustmentApp(root)

    # Start the threads
    app.start_threads()

    root.mainloop()

if __name__ == "__main__":
    main()
 
