import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
import joblib
from tqdm import tqdm  

frames_cluster = 60  
batch_size = 2     

def apply_mask_and_stack_flow(flow_dir, mask_dir, frames_per_cluster=frames_cluster):
    """
    Applique les masques aux flux optiques et regroupe les flux en lots.
    """

    try:
        flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        if len(flow_files) != len(mask_files) - 1:
            raise ValueError("Le nombre de masques doit être supérieur au nombre de flux de 1.")

        temp_stack = []

        for i, flow_file in enumerate(flow_files):
            flow_path = os.path.join(flow_dir, flow_file)
            mask_path = os.path.join(mask_dir, mask_files[i + 1])

            flow = cv2.imread(flow_path).astype(np.float32)
            if flow is None:
                print(f"Warning: Impossible de lire le fichier de flux {flow_path}")
                continue
            flow /= 255.0 

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Impossible de lire le fichier de masque {mask_path}")
                continue
            mask = mask.astype(np.float32) / 255.0

            masked_flow = flow * mask[..., np.newaxis]  
            temp_stack.append(masked_flow.flatten())

            if (i + 1) % frames_per_cluster == 0:
                stacked_group = np.vstack(temp_stack)
                yield stacked_group
                temp_stack = []

        if temp_stack:
            stacked_group = np.vstack(temp_stack)
            yield stacked_group

    except Exception as e:
        print(f"Erreur dans apply_mask_and_stack_flow: {e}")
        return

def process_and_cluster_flows(
    flow_root,
    mask_root,
    output_dir,
    frames_per_cluster=frames_cluster,
    n_clusters=2,
    n_components=50,  
    save_model_path_kmeans="kmeans_model.pkl",
    save_model_path_pca="pca_model.pkl"
):
    """
    Traite les flux optiques bruts avec masques, effectue le clustering en traitant les données en lots.

    Parameters:
    - flow_root: Dossier contenant les flux optiques.
    - mask_root: Dossier contenant les masques.
    - output_dir: Dossier où sauvegarder les résultats.
    - frames_per_cluster: Nombre de frames à regrouper.
    - n_clusters: Nombre de clusters pour K-Means.
    - n_components: Nombre de dimensions après PCA.
    - save_model_path_kmeans: Chemin pour sauvegarder le modèle K-Means entraîné.
    - save_model_path_pca: Chemin pour sauvegarder le modèle PCA entraîné.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Créé le dossier de sortie: {output_dir}")

        flow_dirs = sorted(os.listdir(flow_root))
        mask_dirs = sorted(os.listdir(mask_root))

        if len(flow_dirs) != len(mask_dirs):
            raise ValueError("Le nombre de dossiers de flux et de masques doit correspondre.")

        groups_per_video = {}  
        
        print("Entraînement de IncrementalPCA...")
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

        for flow_dir_name, mask_dir_name in tqdm(zip(flow_dirs, mask_dirs), total=len(flow_dirs), desc="Pass 1: IncrementalPCA"):
            flow_dir = os.path.join(flow_root, flow_dir_name)
            mask_dir = os.path.join(mask_root, mask_dir_name)

            print(f"Processing {flow_dir_name}...")

            group_count = 0
            for group in apply_mask_and_stack_flow(flow_dir, mask_dir, frames_cluster):
                ipca.partial_fit(group)
                group_count += 1

            groups_per_video[flow_dir_name] = group_count
            print(f"Nombre de groupes pour {flow_dir_name}: {group_count}")

        joblib.dump(ipca, save_model_path_pca)
        print(f"IncrementalPCA model saved to {save_model_path_pca}")

        # Entraînement de MiniBatchKMeans
        print("Entraînement de MiniBatchKMeans...")
        mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size)

        for flow_dir_name, mask_dir_name in tqdm(zip(flow_dirs, mask_dirs), total=len(flow_dirs), desc="Pass 2: MiniBatchKMeans"):
            flow_dir = os.path.join(flow_root, flow_dir_name)
            mask_dir = os.path.join(mask_root, mask_dir_name)

            print(f"Processing {flow_dir_name}...")

            for group in apply_mask_and_stack_flow(flow_dir, mask_dir, frames_cluster):
                group_reduced = ipca.transform(group.reshape(1, -1))
                mbk.partial_fit(group_reduced)

        joblib.dump(mbk, save_model_path_kmeans)
        print(f"MiniBatchKMeans model saved to {save_model_path_kmeans}")

        # Assignation des labels et sauvegarde
        print("Assignation des labels...")
        for flow_dir_name, mask_dir_name in tqdm(zip(flow_dirs, mask_dirs), total=len(flow_dirs), desc="Pass 3: Assign Labels"):
            video_output_dir = os.path.join(output_dir, flow_dir_name)
            os.makedirs(video_output_dir, exist_ok=True)

            flow_dir = os.path.join(flow_root, flow_dir_name)
            mask_dir = os.path.join(mask_root, mask_dir_name)

            group_id = 0
            for group in apply_mask_and_stack_flow(flow_dir, mask_dir, frames_cluster):
                group_reshaped = group.reshape(1, -1)

                group_reduced = ipca.transform(group_reshaped)

                label = mbk.predict(group_reduced)[0]

                labels_path = os.path.join(video_output_dir, f"group_{group_id:04d}_label.npy")
                np.save(labels_path, label)
                print(f"Saved label for group {group_id:04d} to {labels_path}")
                group_id += 1

        print("Clustering et assignation des labels terminés.")
    except Exception as e:
        print(f"Une erreur s'est produite dans process_and_cluster_flows: {e}")

if __name__ == "__main__":
    flow_root = "data/flows/Farneback"      
    mask_root = "data/masks1"               
    output_dir = "data/clustering_results" 

    process_and_cluster_flows(
        flow_root=flow_root,
        mask_root=mask_root,
        output_dir=output_dir,
        n_clusters=2,
        n_components=50,  
        frames_per_cluster=frames_cluster,
        save_model_path_kmeans="kmeans_model.pkl",
        save_model_path_pca="pca_model.pkl"
    )

