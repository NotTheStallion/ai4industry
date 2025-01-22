import os
import cv2
import numpy as np
from utils import *

def process_video_with_optical_flow(video_path, start_time, end_time, live_display=True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
        print("Error: Invalid time range.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = None

    video_name = video_path.split('/')[-1].split('.')[0]
    output_path = f"{video_name}_flow.mp4"
    frame_height, frame_width = prev_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    try:
        while cap.isOpened():
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = deepflow_optical_flow(prev_frame, frame, flow)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            print(flow)

            method_optical_flow = "deepflow"
            frame_filename = os.path.join(f"{method_optical_flow}/output_frames", f"frame_{current_frame_idx:04d}.png")
            cv2.imwrite(frame_filename, rgb_flow)

            if live_display:
                cv2.imshow('Optical Flow', rgb_flow)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Process interrupted by user.")
                    break

            prev_frame = frame
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    video_path = "data/test.mp4"
    start_time = 0
    end_time = 3
    process_video_with_optical_flow(video_path, start_time, end_time, live_display=False)
