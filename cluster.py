import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib


frames_cluster= 120


def apply_mask_and_stack_flow(flow_dir, mask_dir, frames_per_cluster=frames_cluster):
    """
    Applique les masques aux flux optiques en s'assurant que chaque flux est aligné
    avec le masque correspondant à la frame de destination.

    Parameters:
    - flow_dir: dossier contenant les flux optiques
    - mask_dir: dossier contenant les masques
    - frames_per_cluster: nombre de frames à regrouper

    Returns:
    - stacked_flows: numpy array contenant les flux empilés pour chaque groupe
    """
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    assert len(flow_files) == len(mask_files) - 1, (
        "Le nombre de masques doit être supérieur au nombre de flux de 1."
    )

    stacked_flows = []
    temp_stack = []

    for i, flow_file in enumerate(flow_files):
        flow_path = os.path.join(flow_dir, flow_file)
        mask_path = os.path.join(mask_dir, mask_files[i + 1])

        flow = cv2.imread(flow_path).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        masked_flow = flow * mask[..., np.newaxis]
        temp_stack.append(masked_flow.flatten())  # Aplatir le flux pour l'empilement

        if (i + 1) % frames_per_cluster == 0:
            stacked_flows.append(np.vstack(temp_stack))  # Empiler les flux d'un groupe
            temp_stack = []

    if temp_stack:
        stacked_flows.append(np.vstack(temp_stack))

    return stacked_flows


def process_and_cluster_flows(flow_root, mask_root, output_dir, frames_per_cluster=20, n_clusters=2, reduce_dim=True, n_components=10, save_model_path="kmeans_model.pkl"):
    """
    Traite les flux optiques bruts avec masques et effectue le clustering.

    Parameters:
    - flow_root: dossier contenant les flux optiques
    - mask_root: dossier contenant les masques
    - output_dir: dossier où sauvegarder les résultats
    - frames_per_cluster: nombre de frames à regrouper
    - n_clusters: nombre de clusters pour K-Means
    - reduce_dim: booléen, si True applique une réduction de dimensionnalité
    - n_components: nombre de dimensions après PCA
    - save_model_path: chemin pour sauvegarder le modèle K-Means entraîné
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flow_dirs = sorted(os.listdir(flow_root))
    mask_dirs = sorted(os.listdir(mask_root))

    assert len(flow_dirs) == len(mask_dirs), "Le nombre de dossiers de flux et de masques doit correspondre."

    all_features = []

    for flow_dir_name, mask_dir_name in zip(flow_dirs, mask_dirs):
        flow_dir = os.path.join(flow_root, flow_dir_name)
        mask_dir = os.path.join(mask_root, mask_dir_name)

        print(f"Processing {flow_dir_name}...")

        stacked_flows = apply_mask_and_stack_flow(flow_dir, mask_dir, frames_per_cluster)

        for group in stacked_flows:
            if reduce_dim:
                pca = PCA(n_components=n_components)
                group = pca.fit_transform(group)

            all_features.append(group)

    all_features = np.vstack(all_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_features)

    joblib.dump(kmeans, save_model_path)
    print(f"K-Means model saved to {save_model_path}")

    start_idx = 0
    for flow_dir_name in flow_dirs:
        video_output_dir = os.path.join(output_dir, flow_dir_name)
        os.makedirs(video_output_dir, exist_ok=True)

        num_groups = len(stacked_flows)
        for group_id in range(num_groups):
            group_labels = kmeans.labels_[start_idx:start_idx + frames_per_cluster]
            labels_path = os.path.join(video_output_dir, f"group_{group_id:04d}_labels.npy")
            np.save(labels_path, group_labels)
            start_idx += frames_per_cluster
            print(f"Saved labels for group {group_id} to {labels_path}")


if __name__ == "__main__":
    flow_root = "data/flows" 
    mask_root = "data/masks"  
    output_dir = "data/clustering_results"  

    process_and_cluster_flows(
        flow_root=flow_root,
        mask_root=mask_root,
        output_dir=output_dir, 
        n_clusters=2,         
        reduce_dim=True,      
        n_components=20       
    )
