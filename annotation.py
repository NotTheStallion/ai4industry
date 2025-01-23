import os
import json

# Chemin vers le dossier contenant les frames
frames_folder = "/home/cbanide/ai4industry/test_flows/farneback_optical_flow"

# Labels pour les annotations
label_first_half = "normal"
label_second_half = "ill"

# Fréquence des frames (par exemple, 30 fps)
frame_rate = 30

# Charger et trier les frames
frames = sorted(os.listdir(frames_folder))  # Trier pour garantir l'ordre temporel

# Calculer le seuil de séparation entre les deux labels
num_frames = len(frames)
half_frame = num_frames // 2  # Séparer en deux périodes égales

# Générer les annotations
annotations = {}
for idx, frame_name in enumerate(frames):
    if idx < half_frame:
        annotations[frame_name] = label_first_half
    else:
        annotations[frame_name] = label_second_half

# Sauvegarder les annotations dans un fichier JSON
annotations_file = os.path.join(frames_folder, "annotations.json")
with open(annotations_file, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations générées et sauvegardées dans {annotations_file}")
