# Digital Signal Processing - Group 8
# Project title: License plate recognition

# Group members: Bence Toth
#				 Davis Zvigulis
#				 Laszlo Balazs Orban

# Importing necessary libraries
from ultralytics import YOLO
from tkinter.filedialog import askopenfilename
from platform import system
from os import getcwd
from sys import exit
import cv2
import easyocr
import torch
import subprocess
import numpy as np
import tkinter as tk

# Tkinter window setup configuration and other config variables
icon_default : str = "icon.ico"
icon_nvidia : str = "icon_nvidia.ico"
title : str = "License plate recognition"
size : str = "500x265"
spacing : str = "+500+200"
font : str = "Arial 16"
background : str = "gray25"
foreground : str = "AntiqueWhite2"
model_path : str = ""
filename : str = ""
error_msg : str = ""
capture_running : bool = False # This variable is used to limit the running capture instance to only one, as you can click the event button multiple times
use_gpu : bool = False # This variable is used to determine if the process runs on CPU (default) or GPU

# Tkinter main window declaration
win = tk.Tk()
win.geometry(f"{size}{spacing}")
win.overrideredirect(False)
win.config(bg = background)
win.resizable(False, False)

# Deciding if the process should run on GPU or CPU (CPU default)
try:
    subprocess.check_output("nvidia-smi") # Check if there is an Nvidia GPU driver installed
    if torch.cuda.is_available(): # Nvidia GPU is detected and CUDA is available, default to CUDA execution
        use_gpu = True
except Exception: # Nvidia GPU not detected, default to CPU execution
    # If there is no Nvidia GPU in the system, the previous subprocess call will fail
    use_gpu = False

# Initialize EasyOCR module with English language selected
reader = easyocr.Reader(['en'], gpu = use_gpu)

# Try to place the small icon on the Tkinter window based on if the CPU or GPU is used for OCR (icon placement works only on Windows for some reason)
if system() == "Windows":
    try :
        gpu_title_text : str = "CUDA" if use_gpu else "CPU"
        win.iconbitmap("r", icon_nvidia if use_gpu else icon_default)
        win.title(f"{title} - using {gpu_title_text}")
    except tk.TclError:
        pass

# This function is used to get the full PATH of the YOLO model file
def select_file(arg = None) -> None :
    global file_label, model_path, filename, capture_running

    if not capture_running:
        model_path = askopenfilename(initialdir = getcwd(), title = "Select File", filetypes = (("PT file", "*.pt"), ("all files", "*.*")))
        filename = model_path.split( "/" )[-1:][0]
        file_label.config(text = f"Chosen file: {filename}")
    return

# This function is used to exit the main Tkinter window on ESC button
def exit_function(arg = None) -> None:
    win.destroy()
    exit()

# This function is used to capture and evaluate the licence plate recognition in real-time
def capture(arg = None) -> None:
    global msg_label, model_path, capture_running

    if not capture_running:
        capture_running = True # Raising flag, start of capture
    else:
        return

    recognized_plates : set = set() # Set to store license plate numbers
    clean_text : str = "" # Stores text from OCR result

    model = YOLO(model_path) # Load YOLO model from path, preferably running on CUDA instead of CPU
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    cap = cv2.VideoCapture(0) # Open webcam

    if not cap.isOpened(): # If there is an error with the webcam, enter clause
        msg_label.config(text = "MSG: Error, could not open webcam") # Printout to panel
        cap.release() # Release webcam
        cv2.destroyAllWindows() # Closing windows
        capture_running = False # Stopping capture
        return
    else: # Continuous frame capture if there is no error with the webcam
        msg_label.config(text = "MSG: Starting capture") # Printout to panel
        while True:
            ret, frame = cap.read()
            if not ret: # An error can happen during the capture
                msg_label.config(text = "MSG: Error during capture") # Printout to panel
                cap.release() # Release webcam
                cv2.destroyAllWindows() # Closing windows
                capture_running = False # Stopping capture
                break
            
            # Acquire the webcam footage
            try:
                results = model(frame)
            except Exception: # If the specified .pt file is not correct, enter clause
                msg_label.config(text = "MSG: Error with specified file") # Printout to panel
                cap.release() # Release webcam
                cv2.destroyAllWindows() # Closing windows
                capture_running = False # Stopping capture
                break

            # Annotate the frame with bounding boxes
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Real-Time Detection", annotated_frame)

            # Process the acquired webcam footage
            for result in results:
                for box in result.boxes: # Each detected object has a bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get top-left and bottom-right coordinates

                    # Crop and preprocess (grayscale)
                    plate_box = frame[y1:y2, x1:x2] # Crop the detected plate region from the frame
                    gray_plate = cv2.cvtColor(plate_box, cv2.COLOR_BGR2GRAY) # Convert the cropped part to grayscale

                    # Displaying cropped and cropped + gray scaled plate
                    cv2.imshow("Cropped Plate (Color)", plate_box)
                    cv2.imshow("Cropped Plate (Grayscale)", gray_plate)

                    # Use EasyOCR on the grayscale transformed crop
                    ocr_results = reader.readtext(gray_plate)

                    for (bbox, text, prob) in ocr_results:
                        clean_text = text.strip() # Remove any leftover whitespace

                        # If new plate detected
                        if clean_text and clean_text not in recognized_plates:
                            recognized_plates.add(clean_text) # Add to the set of recognized plates

                        # Draw main bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw white-filled box above the plate
                        (text_width, text_height), _ = cv2.getTextSize(clean_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2) # Determine size of text
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (255, 255, 255), -1) # Draw white box

                        # Put text inside that white box
                        cv2.putText(frame, clean_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 0), 2)

            # Display detection and annotation window
            cv2.imshow("License Plate Detection + OCR", frame)

            # Display recognized plate number
            plate_display = 255 * np.ones((150, 600, 3), dtype=np.uint8)

            # Write last recognized plate number
            cv2.putText(plate_display, f"Plate number: {clean_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.imshow("Recognized Plate", plate_display) # Show number in separate window

            # Exit the live capture in the event of keyboard press of button 'Q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                # Cleanup
                msg_label.config(text = "MSG: closed real-time capture") # Printout to panel
                cap.release() # Release webcam
                cv2.destroyAllWindows() # Closing windows
                capture_running = False # Stopping capture
                break
    
# Declaration of used Tkinter widgets

filename_in = tk.Button(win, text = "Select model file", bg = background, fg = foreground, bd = 1, anchor = "center", activebackground = foreground, activeforeground = background, font = font, command = select_file)
filename_in.place ( x = 5 , y = 5 , width = 480 , height = 60 )

file_label = tk.Label(win, text = f"Chosen file: {filename}", bg = background, fg = foreground, anchor = "center", font = font)
file_label.place ( x = 5 , y = 70 , width = 480 , height = 60 )

msg_label = tk.Label(win, text = f"{error_msg}", bg = background, fg = foreground, anchor = "center", font = font)
msg_label.place ( x = 5 , y = 135 , width = 480 , height = 60 )

capture_start_button = tk.Button(win, text = "Start video capture", bg = background, fg = foreground, bd = 1, anchor = "center", activebackground = foreground, activeforeground = background, font = font, command = capture)
capture_start_button.place ( x = 5 , y = 200 , width = 480 , height = 60 )

# Binding the declared buttons to their corresponding functions
win.bind("<Escape>", exit_function) # Exit on ESC button press
win.bind("<Return>", capture) # Start capture function on ENTER button

win.mainloop()