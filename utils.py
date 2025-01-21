import cv2 as cv

def farneback_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using Farneback method."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        gray1, gray2, prev_flow, pyr_scale=0.5, levels=5, winsize=50, iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    return flow

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