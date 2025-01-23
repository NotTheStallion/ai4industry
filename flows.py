import cv2
import numpy as np
import os

def farneback_optical_flow(prev_gray, gray):
    """
    Calcule le flux optique en utilisant la méthode de Farnebäck.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow

def deepflow_optical_flow(prev_gray, gray):
    """
    Calcule le flux optique en utilisant la méthode DeepFlow.
    """
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(prev_gray, gray, None)
    return flow

def process_video_with_optical_flow(video_path, output_dir, optical_flow_method, method_name, display_flow=True):
    """
    Traite une vidéo, calcule les flux optiques pour toute la vidéo et les sauvegarde dans un dossier spécifique.

    Parameters:
    - video_path: chemin vers la vidéo
    - output_dir: dossier de sortie pour les flux optiques
    - optical_flow_method: méthode pour calculer le flux optique (fonction)
    - method_name: nom de la méthode (pour créer un dossier spécifique)
    - display_flow: booléen, afficher ou non les flux optiques
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    method_dir = os.path.join(output_dir, method_name)
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)

    flow_dir = os.path.join(method_dir, video_name + "_flows")
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of {video_path}.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0

    frame_total_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break
        frame_total_count += 1
        print(f"Processing frame: {frame_total_count}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = optical_flow_method(prev_gray, gray)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        flow_filename = os.path.join(flow_dir, f"flow_{frame_count:04d}.png")
        cv2.imwrite(flow_filename, rgb_flow)
        frame_count += 1

        if display_flow:
            cv2.imshow('Optical Flow', rgb_flow)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

def process_video_folder(input_folder, output_folder, display_flow=False):
    """
    Parcourt un dossier de vidéos, calcule les flux optiques avec plusieurs méthodes pour chacune et les sauvegarde.

    Parameters:
    - input_folder: dossier contenant les vidéos
    - output_folder: dossier où sauvegarder les flux optiques
    - display_flow: booléen, afficher ou non les flux optiques
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4','.MP4' ,'.avi'))]
    methods = {
        "Farneback": farneback_optical_flow
        # "DeepFlow": deepflow_optical_flow
    }

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing video: {video_file}")
        for method_name, method in methods.items():
            print(f"\tUsing method: {method_name}")
            process_video_with_optical_flow(
                video_path, output_folder, method, method_name, display_flow
            )

if __name__ == "__main__":
    input_folder = "data/"  
    output_folder = "data/flows"
    display_flow = False

    process_video_folder(input_folder, output_folder, display_flow)


















# import cv2
# import numpy as np
# import os

# def farneback_optical_flow(prev_gray, gray):
#     """
#     Calcule le flux optique en utilisant la méthode de Farnebäck.
#     """
#     flow = cv2.calcOpticalFlowFarneback(
#         prev_gray, gray, None,
#         pyr_scale=0.5,
#         levels=3,
#         winsize=15,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0
#     )
#     return flow

# def deepflow_optical_flow(prev_gray, gray):
#     """
#     Calcule le flux optique en utilisant la méthode DeepFlow.
#     """
#     deepflow = cv2.optflow.createOptFlow_DeepFlow()
#     flow = deepflow.calc(prev_gray, gray, None)
#     return flow

# def process_video_with_optical_flow(video_path, start_time, end_time, output_dir, optical_flow_method, method_name, display_flow=True):
#     """
#     Traite une vidéo, calcule les flux optiques pour une plage de temps et les sauvegarde dans un dossier spécifique.

#     Parameters:
#     - video_path: chemin vers la vidéo
#     - start_time: temps de début en secondes
#     - end_time: temps de fin en secondes
#     - output_dir: dossier de sortie pour les flux optiques
#     - optical_flow_method: méthode pour calculer le flux optique (fonction)
#     - method_name: nom de la méthode (pour créer un dossier spécifique)
#     - display_flow: booléen, afficher ou non les flux optiques
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}.")
#         return

#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     method_dir = os.path.join(output_dir, method_name)
#     if not os.path.exists(method_dir):
#         os.makedirs(method_dir)

#     flow_dir = os.path.join(method_dir, video_name + "_flows")
#     if not os.path.exists(flow_dir):
#         os.makedirs(flow_dir)

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     start_frame = int(start_time * fps)
#     end_frame = int(end_time * fps)

#     if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
#         print(f"Error: Invalid frame range for video {video_path}.")
#         return

#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     ret, prev_frame = cap.read()
#     if not ret:
#         print(f"Error: Could not read the first frame of {video_path}.")
#         return

#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     frame_count = 0

#     while cap.isOpened():
#         current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         if current_frame_idx > end_frame:
#             break

#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         flow = optical_flow_method(prev_gray, gray)

#         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         hsv = np.zeros_like(frame)
#         hsv[..., 1] = 255
#         hsv[..., 0] = angle * 180 / np.pi / 2
#         hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#         rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#         flow_filename = os.path.join(flow_dir, f"flow_{frame_count:04d}.png")
#         cv2.imwrite(flow_filename, rgb_flow)
#         frame_count += 1

#         if display_flow:
#             cv2.imshow('Optical Flow', rgb_flow)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         prev_gray = gray

#     cap.release()
#     cv2.destroyAllWindows()

# def process_video_folder(input_folder, output_folder, start_time, end_time, display_flow=False):
#     """
#     Parcourt un dossier de vidéos, calcule les flux optiques avec plusieurs méthodes pour chacune et les sauvegarde.

#     Parameters:
#     - input_folder: dossier contenant les vidéos
#     - output_folder: dossier où sauvegarder les flux optiques
#     - start_time: temps de début en secondes
#     - end_time: temps de fin en secondes
#     - display_flow: booléen, afficher ou non les flux optiques
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi'))]
#     methods = {
#         "Farneback": farneback_optical_flow,
#         "DeepFlow": deepflow_optical_flow
#     }

#     for video_file in video_files:
#         video_path = os.path.join(input_folder, video_file)
#         print(f"Processing video: {video_file}")
#         for method_name, method in methods.items():
#             print(f"\tUsing method: {method_name}")
#             process_video_with_optical_flow(
#                 video_path, start_time, end_time, output_folder, method, method_name, display_flow
#             )

# if __name__ == "__main__":
#     input_folder = "data"  # Dossier contenant les vidéos
#     output_folder = "data/flows"  # Dossier où sauvegarder les flux
#     start_time = 12  # Temps de début en secondes
#     end_time = 14    # Temps de fin en secondes
#     display_flow = False

#     process_video_folder(input_folder, output_folder, start_time, end_time, display_flow)
