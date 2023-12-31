import threading
import cv2
from ultralytics import YOLO
import os

# Prevents Runtime error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def run_tracker_in_thread(filename, model, file_index):
    video = cv2.VideoCapture(filename)

    while True:
        ret, frame = video.read()

        if not ret:
            break
    
        results = model.track(frame, persist = True)
        res_plotted = results[0].plot()
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()

# Load the Models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')

# Define the cideo files for the trackers
video_file1 = "traffic3.mp4" # Path to video file, 0 for webcam
video_file2 = 0 # Path to video file, 0 for webcam, 1 for external camera

# Create tracker threads
tracker_thread1 = threading.Thread(target = run_tracker_in_thread, args = (video_file1, model1, 1), daemon = True)
tracker_thread2 = threading.Thread(target = run_tracker_in_thread, args = (video_file2, model2, 2), daemon = True)

# Start the Tracker Threads
tracker_thread1.start()
tracker_thread2.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
