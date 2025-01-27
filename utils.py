import cv2 as cv
import os
import numpy as np


def farneback_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using Farneback method."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        gray1, gray2, prev_flow, pyr_scale=0.5, levels=5, winsize=50, iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    return flow

def cuda_farneback_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using Farneback method on GPU."""
    gray1 = cv.cuda_GpuMat()
    gray1.upload(cv.cvtColor(frame1, cv.COLOR_BGR2GRAY))
    gray2 = cv.cuda_GpuMat()
    gray2.upload(cv.cvtColor(frame2, cv.COLOR_BGR2GRAY))
    flow = cv.cuda.FarnebackOpticalFlow.create(
        0.5, 5, 50, 3, 7, 1.5, 0
    ).calc(gray1, gray2, prev_flow)
    return flow.download()

def pcaflow_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using PCAFlow."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    pcaflow = cv.optflow.createOptFlow_PCAFlow()
    flow = pcaflow.calc(gray1, gray2, prev_flow)
    return flow

# @critical : error display
def simpleflow_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using SimpleFlow."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    simpleflow = cv.optflow.createOptFlow_SimpleFlow()
    flow = simpleflow.calc(gray1, gray2, prev_flow)
    return flow

def deepflow_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using DeepFlow."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    deepflow = cv.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(gray1, gray2, prev_flow)
    return flow

# @critical : not working
def dis_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using DIS (Dense Inverse Search)."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    dis = cv.optflow.createOptFlow_DIS(cv.optflow.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(gray1, gray2, prev_flow)
    return flow



def progress_video(path="test_flows/deepflow_optical_flow/",name="test.mp4"):
    """
    Create a video from the images saved in the steps/ folder."""
    out_video_name = "temp_nca.mp4"
    out_video_full_path = out_video_name

    pre_imgs = os.listdir(path)
    pre_imgs.sort()
    img = []
    for i in pre_imgs:
        i = path + i
        img.append(i)

    cv2_fourcc = cv.VideoWriter_fourcc(*"mp4v")

    frame = cv.imread(img[0])

    size = list(frame.shape)
    del size[2]
    size.reverse()

    video = cv.VideoWriter(
        out_video_full_path, cv2_fourcc, 24, size
    )  # output video name, fourcc, fps, size

    for i in range(len(img)):
        video.write(cv.imread(img[i]))

    video.release()
    os.system(f"ffmpeg -y -i temp_nca.mp4 {name} -loglevel quiet")
    os.system("rm -f temp_nca.mp4")


def read_frames(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".jpg", ".png")):
            img = cv.imread(os.path.join(directory, filename))
            if img is not None:
                frames.append(img)
    return frames


def read_flow(filename):
        flow = cv.readOpticalFlow(filename)
        if flow is None:
            return None
        flow = flow[..., :2]
        return flow


def read_flows(directory):
    flows = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".flo"):
            flow = read_flow(os.path.join(directory, filename))
            if flow is not None:
                flows.append(flow)
    return flows


def write_flow(flow, filename):
    cv.writeOpticalFlow(filename, flow)


def flow2bgr(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # cv.imshow('Optical Flow', bgr)
    return bgr


def compute_ae(flow1, flow2):
    return np.mean(np.abs(flow1 - flow2))


def compute_epe(flow1, flow2):
    return np.mean(np.linalg.norm(flow1 - flow2, axis=-1))


def compute_mse(frame1, frame2):
    return np.mean((frame1 - frame2) ** 2)


def project(frame, flow):
    h, w = frame.shape[:2]
    flow_map = np.zeros_like(frame)

    for y in range(h):
        for x in range(w):
            dx, dy = flow[y, x]
            new_x = int(x + dx)
            new_y = int(y + dy)

            if 0 <= new_x < w and 0 <= new_y < h:
                flow_map[new_y, new_x] = frame[y, x]

    return flow_map


if __name__=="__main__":
    progress_video(path="test_flows/farneback_optical_flow/",name="test_flows_DPFLW.mp4")