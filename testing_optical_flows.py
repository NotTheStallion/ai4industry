import cv2
import numpy as np
import os
from utils import *
from flow_viz import *
import wandb

def process_video_with_optical_flow(video_path, start_time, end_time, flow_method=farneback_optical_flow, display_flow=False):
    wandb_run = wandb.init(project="ai4i", name=flow_method.__name__)
    wandb.config.update({"dir": video_path, "method": flow_method.__name__})   
    wandb.define_metric("frame")
    wandb.define_metric("*", step_metric="frame")
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    flow_dir = video_name + "_flows"
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
        print("Error: Invalid frame range.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while cap.isOpened():
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame_idx > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        print("Error: No frames were read.")
        return

    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = None
    frame_count = 0

    for i, frame in enumerate(frames[1:]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = flow_method(prev_frame, frame, flow)
        
        prev_frame = frame

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # rgb_flow = flow_to_image(flow, convert_to_bgr=True)
        
        compensated_frame = project(prev_frame, flow)
        e_mse = compute_mse(compensated_frame, frame)
        
        wandb.log({"frame": (i/len(frames))*100, "MSE": e_mse})

        method_dir = os.path.join(flow_dir, flow_method.__name__)
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)
        flow_filename = os.path.join(method_dir, f"flow_{frame_count:04d}.png")
        cv2.imwrite(flow_filename, rgb_flow)
        frame_count += 1

        if display_flow:
            cv2.imshow('Optical Flow', rgb_flow)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    wandb_run.finish()

if __name__ == "__main__":
    video_path = "data/test.mp4"
    start_time = 12
    end_time = 14
    display_flow = False

    process_video_with_optical_flow(video_path, start_time, end_time, flow_method=farneback_optical_flow)
