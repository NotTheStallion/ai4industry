import cv2 as cv
import os



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


if __name__=="__main__":
    progress_video(path="test_flows/farneback_optical_flow/",name="test_flows_FrnBk.mp4")