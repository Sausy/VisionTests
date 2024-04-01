from openni import openni2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

class ColorAdjustmentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Color Adjustment App")

        self.color_correction_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.create_widgets()

    def create_widgets(self):
        # Color Adjustment Matrix Sliders
        self.sliders = []
        for i in range(3):
            for j in range(3):
                slider_label = ttk.Label(self.master, text=f"M[{i}][{j}]")
                slider_label.grid(row=i, column=j, padx=5, pady=5)

                slider_var = tk.DoubleVar()
                slider_var.set(self.color_correction_matrix[i, j])
                slider = ttk.Scale(self.master, from_=0.0, to=2.0, length=200, variable=slider_var, orient=tk.HORIZONTAL)
                slider.grid(row=i, column=j + 1, padx=5, pady=5)

                slider_var.trace_add("write", self.update_matrix)
                self.sliders.append(slider_var)

        # OpenNI Initialization
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        self.color_stream = self.dev.create_color_stream()
        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream.start()
        self.depth_stream.start()

        # Display Frames Button
        self.display_button = ttk.Button(self.master, text="Display Frames", command=self.display_frames)
        self.display_button.grid(row=3, column=0, columnspan=4, pady=10)

    def update_matrix(self, *args):
        # Update color correction matrix based on slider values
        for i in range(3):
            for j in range(3):
                self.color_correction_matrix[i, j] = self.sliders[i * 3 + j].get()

    def display_frames(self):
        try:
            while True:
                color_frame = self.color_stream.read_frame()
                depth_frame = self.depth_stream.read_frame()

                color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
                color_image = color_data.reshape((color_frame.height, color_frame.width, 3))

                # Apply color correction matrix
                color_image_corrected = cv2.transform(color_image, self.color_correction_matrix)

                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
                depth_image = depth_data.reshape((depth_frame.height, depth_frame.width))

                normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

                cv2.imshow('Color Frame', color_image_corrected)
                cv2.imshow('Depth Frame', depth_colormap)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.color_stream.stop()
            self.depth_stream.stop()
            openni2.unload()
            cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    app = ColorAdjustmentApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
from openni import openni2
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

class ColorAdjustmentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Color Adjustment App")

        self.color_correction_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.create_widgets()

    def create_widgets(self):
        # Color Adjustment Matrix Sliders
        self.sliders = []
        for i in range(3):
            for j in range(3):
                slider_label = ttk.Label(self.master, text=f"M[{i}][{j}]")
                slider_label.grid(row=i, column=j, padx=5, pady=5)

                slider_var = tk.DoubleVar()
                slider_var.set(self.color_correction_matrix[i, j])
                slider = ttk.Scale(self.master, from_=0.0, to=2.0, length=200, variable=slider_var, orient=tk.HORIZONTAL)
                slider.grid(row=i, column=j + 1, padx=5, pady=5)

                slider_var.trace_add("write", self.update_matrix)
                self.sliders.append(slider_var)

        # OpenNI Initialization
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        self.color_stream = self.dev.create_color_stream()
        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream.start()
        self.depth_stream.start()

        # Display Frames Button
        self.display_button = ttk.Button(self.master, text="Display Frames", command=self.display_frames)
        self.display_button.grid(row=3, column=0, columnspan=4, pady=10)

    def update_matrix(self, *args):
        # Update color correction matrix based on slider values
        for i in range(3):
            for j in range(3):
                self.color_correction_matrix[i, j] = self.sliders[i * 3 + j].get()

    def display_frames(self):
        try:
            while True:
                color_frame = self.color_stream.read_frame()
                depth_frame = self.depth_stream.read_frame()

                color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
                color_image = color_data.reshape((color_frame.height, color_frame.width, 3))

                # Apply color correction matrix
                color_image_corrected = cv2.transform(color_image, self.color_correction_matrix)

                depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
                depth_image = depth_data.reshape((depth_frame.height, depth_frame.width))

                normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

                cv2.imshow('Color Frame', color_image_corrected)
                cv2.imshow('Depth Frame', depth_colormap)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.color_stream.stop()
            self.depth_stream.stop()
            openni2.unload()
            cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    app = ColorAdjustmentApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
 
