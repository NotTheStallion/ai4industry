import cv2
import numpy as np
from utils import *



def process_video_with_optical_flow(video_path, start_time, end_time):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the video frame rate
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Validate frame range
    if start_frame >= total_frames or end_frame > total_frames:
        print("Error: Specified times are out of video range.")
        return

    if start_frame >= end_frame:
        print("Error: Start time must be less than end time.")
        return

    # Set the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flow = None
    while cap.isOpened():
        # Get the current frame index
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame_idx > end_frame:
            break

        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Farneback optical flow
        flow = deepflow_optical_flow(prev_frame, frame, flow)

        # Visualize the flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show the optical flow visualization
        cv2.imshow('Optical Flow', rgb_flow)

        # Update the previous frame
        prev_gray = gray

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/test.MP4"
    start_time = 350
    end_time = 400

    process_video_with_optical_flow(video_path, start_time, end_time)