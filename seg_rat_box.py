import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from ultralytics import YOLO

# Configuration YOLO
model = YOLO('yolo11s-seg.pt')
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 100
LOW_RES = (320, 180)

# Configuration SAM
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

def detect_mouse_color(frame):
    """
    Détecte la souris en se basant sur sa couleur dominante (marron).
    """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([5, 50, 50])  
    upper_bound = np.array([20, 255, 255])
    
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, x + w, y + h 
    return None  

def detect_and_draw(frame, frame_count, output_mask_dir, previous_mask=None):
    """
    Crée un masque pour chaque frame. Si aucune détection via YOLO ou SAM,
    utilise la méthode basée sur la couleur de la souris.
    """
    low_res_frame = cv2.resize(frame, LOW_RES)
    results = model(low_res_frame, verbose=False)
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    largest_area = 0
    largest_detection = None

    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_detection = (x1, y1, x2, y2, conf, cls)

    if largest_detection:
        x1, y1, x2, y2, conf, cls = largest_detection
        print(f"Detection found for frame {frame_count} via YOLO.")
    else:
        print(f"No detection for frame {frame_count} via YOLO. Trying color-based detection.")
        mouse_color_box = detect_mouse_color(frame)
        if mouse_color_box:
            x1, y1, x2, y2 = mouse_color_box
            print(f"Detection found for frame {frame_count} via color.")
        else:
            if previous_mask is not None:
                print(f"No detection for frame {frame_count} via color. Using previous mask.")
                return previous_mask
            else:
                print(f"No detection for frame {frame_count} via any method. Generating empty mask.")
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                mask_filename = os.path.join(output_mask_dir, f"mask_{frame_count:04d}.png")
                cv2.imwrite(mask_filename, mask)
                return mask

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask_filename = os.path.join(output_mask_dir, f"mask_{frame_count:04d}.png")
    cv2.imwrite(mask_filename, mask)
    print(f"Saved mask: {mask_filename}")

    return mask

def process_videos(input_folder, output_folder):
    """
    Traite chaque vidéo du dossier d'entrée, génère des masques pour chaque frame, et les sauvegarde.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.MP4', '.avi'))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        previous_mask = None 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            previous_mask = detect_and_draw(frame, frame_count, video_output_dir, previous_mask)
            frame_count += 1

        cap.release()
        print(f"Finished processing video: {video_file}")

if __name__ == "__main__":
    input_folder = "./data" 
    output_folder = "data/masks1"  

    if torch.cuda.is_available():
        model.to('cuda')

    process_videos(input_folder, output_folder)
