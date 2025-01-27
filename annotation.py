import os
import json

# dossier de frames
frames_folder = "/home/cbanide/ai4industry/test_flows/farneback_optical_flow"

# Labels 
label_first_half = "normal"
label_second_half = "ill"

# FPS
frame_rate = 30

frames = sorted(os.listdir(frames_folder))  

num_frames = len(frames)
half_frame = num_frames // 2  

# annotations
annotations = {}
for idx, frame_name in enumerate(frames):
    if idx < half_frame:
        annotations[frame_name] = label_first_half
    else:
        annotations[frame_name] = label_second_half


annotations_file = os.path.join(frames_folder, "annotations.json")
with open(annotations_file, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations générées et sauvegardées dans {annotations_file}")
