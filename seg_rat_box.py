import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from ultralytics import YOLO

model = YOLO('yolo11s-seg.pt')
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 100
LOW_RES = (320, 180)

CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")



def generate_default_rectangle(frame):
    """
    Génère un rectangle par défaut au centre de l'image si aucune détection n'est disponible.
    """
    rect_width = 350  # Largeur du rectangle
    rect_height = 350  # Hauteur du rectangle

    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    rect_x1 = max(0, center_x - rect_width // 2)
    rect_y1 = max(0, center_y - rect_height // 2)
    rect_x2 = min(frame_width, center_x + rect_width // 2)
    rect_y2 = min(frame_height, center_y + rect_height // 2)

    return rect_x1, rect_y1, rect_x2, rect_y2

def detect_and_draw(frame, frame_count, output_mask_dir):
    """
    Crée un masque pour chaque frame où la souris est détectée et sauvegarde ce masque.
    """
    low_res_frame = cv2.resize(frame, LOW_RES)
    results = model(low_res_frame, verbose=False)
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    rect_width = 350 
    rect_height = 350 

    largest_area = 0
    largest_detection = None

    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_detection = (x1, y1, x2, y2, conf, cls)

    # if largest_detection:
    #     x1, y1, x2, y2, conf, cls = largest_detection
    #     center_x = (x1 + x2) // 2
    #     center_y = (y1 + y2) // 2

    #     rect_x1 = max(0, center_x - rect_width // 2)
    #     rect_y1 = max(0, center_y - rect_height // 2)
    #     rect_x2 = min(frame.shape[1], center_x + rect_width // 2)
    #     rect_y2 = min(frame.shape[0], center_y + rect_height // 2)

    #     mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    #     cv2.rectangle(mask, (rect_x1, rect_y1), (rect_x2, rect_y2), 255, -1)
    if largest_detection:
        x1, y1, x2, y2, conf, cls = largest_detection
        print(f"Detection found for frame {frame_count}.")
    else:
        print(f"No detection for frame {frame_count}. Using default region.")
        x1, y1, x2, y2 = generate_default_rectangle(frame)

    # Créer le masque
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    mask_filename = os.path.join(output_mask_dir, f"mask_{frame_count:04d}.png")
    cv2.imwrite(mask_filename, mask)
    print(f"Saved mask: {mask_filename}")

def process_videos(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4','.MP4', '.avi'))]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detect_and_draw(frame, frame_count, video_output_dir)

            frame_count += 1

        cap.release()
        print(f"Finished processing video: {video_file}")

if __name__ == "__main__":
    input_folder = "./data" 
    output_folder = "data/masks"  

    if torch.cuda.is_available():
        model.to('cuda')

    process_videos(input_folder, output_folder)
