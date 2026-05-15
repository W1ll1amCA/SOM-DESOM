import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mode



def flatten_weights(weights: np.ndarray) -> np.ndarray:
    """
    Transforme les poids du SOM en forme (n_nodes, dim).

    Accepte :
    - (n_nodes, dim)
    - (rows, cols, dim)
    """
    weights = np.asarray(weights)

    if weights.ndim == 3:
        return weights.reshape(-1, weights.shape[-1])

    if weights.ndim == 2:
        return weights

    raise ValueError(f"Format de poids invalide : {weights.shape}. "
                    "Attendu : (n_nodes, dim) ou (rows, cols, dim).")

# =========================
# BMU
# =========================

def compute_BMU(data : np.ndarray, poids : np.ndarray) -> np.ndarray:
    """
    Calcule le BMU pour chaque point de données
    Args:
        data (np.ndarray): Données d'entrée de forme (n_samples, input_dim)
        poids (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
    Returns:
        bmu_indices: np.ndarray de forme (n_samples,)
        distances: np.ndarray de forme (n_samples,)
    """
    poids = flatten_weights(poids)
    data = np.asarray(data, dtype=float)
    
    dist = np.linalg.norm(data[:, np.newaxis] - poids, axis=2)

    bmu_indices = np.argmin(dist, axis=1)
    distances = np.min(dist, axis=1)

    return bmu_indices, distances


# =============================================================================================================================
# METRIQUES
# =============================================================================================================================

def err_quantification(data : np.ndarray, poids : np.ndarray) -> float:
    """
    Calcule l'erreur de quantification du SOM
    Args:
        data (np.ndarray): Données d'entrée de forme (n_samples, input_dim)
        poids (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
    Returns:
        float: Erreur de quantification moyenne
    """
    _, distances = compute_BMU(data, poids)
    return np.mean(distances)

def err_topologique(data : np.ndarray, poids : np.ndarray, grid_size : tuple) -> float:
    """
    Calcule l'erreur topologique du SOM
    Args:
        data (np.ndarray): Données d'entrée de forme (n_samples, input_dim)
        poids (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
        grid_size (tuple): Taille de la grille du SOM
    Returns:
        float: Erreur topologique moyenne
    """
    data = np.asarray(data, dtype=float)
    poids = flatten_weights(poids)

    r,c = grid_size

    if poids.shape[0] != r * c:
        raise ValueError(f"Incohérence entre grid_size={grid_size} et le nombre de poids={poids.shape[0]}.")

    dist = np.linalg.norm(data[:, np.newaxis] - poids, axis=2)
    
    les_2_best = np.argsort(dist, axis=1)[:, :2]

    errors = 0

    for bmu1,bmu2 in les_2_best:
        r1, c1 = divmod(bmu1, c)
        r2, c2 = divmod(bmu2, c)
        
        # distance de Manhattan entre les positions des deux BMU sur la grille
        grid_dist = abs(r1 - r2) + abs(c1 - c2)

        if grid_dist > 1:
            errors += 1
    
    return errors / len(data)

def purity_sc(y_true : np.ndarray, bmu_idx : np.ndarray) -> float:
    """
    Calcule la pureté d'un clustering
    Args:
        y_true (np.ndarray): Labels vrais de forme (n_samples,)
        bmu_idx (np.ndarray): Indices des BMU de forme (n_samples,)
    Returns:
        float: Pureté du clustering
    """
    tot_correct = 0
    clst = np.unique(bmu_idx)

    for c in clst:
        labels_in_clst = y_true[bmu_idx == c]

        if len(labels_in_clst) == 0:
            continue

        maj_label = mode(labels_in_clst, keepdims=True).mode[0]
        tot_correct += np.sum(labels_in_clst == maj_label)

    return tot_correct / len(y_true)

def nmi_sc(y_true : np.ndarray, bmu_idx : np.ndarray) -> float:
    """
    Calcule le score de Normalized Mutual Information (NMI) pour évaluer la qualité du clustering
    Args:
        y_true (np.ndarray): Labels vrais de forme (n_samples,)
        bmu_idx (np.ndarray): Indices des BMU de forme (n_samples,)
    Returns:
        float: Score NMI du clustering
    """
    return normalized_mutual_info_score(y_true, bmu_idx)


# =============================================================================================================================
# TABLEAU DES RESULTATS
# =============================================================================================================================

def evaluate_som_model(model_name : str, data : np.ndarray, 
                       weights : np.ndarray, y_true : np.ndarray, 
                       grid_size : tuple, training_time : float = None)-> dict:
    """
    Évaluation complete d'un modèle.

    Args:
        model_name (str): nom du modèle, par exemple "SOM", "AE+SOM", "DESOM"
        data (np.ndarray): données utilisées pour l'évaluation
              - SOM classique : données originales
              - AE+SOM : embeddings latents
              - DESOM : embeddings latents
        weights (np.ndarray): prototypes du SOM
        y_true (np.ndarray): labels réels
        grid_size (tuple): taille de la grille
        training_time (float, optional): temps d'entraînement en secondes

    Returns:
        dict contenant les métriques
    """

    weights = flatten_weights(weights)

    bmu_indices, _ = compute_BMU(data, weights)

    results = {
        "Modèle": model_name,
        "QE": err_quantification(data, weights),
        "TE": err_topologique(data, weights, grid_size),
        "Pureté": purity_sc(y_true, bmu_indices),
        "NMI": nmi_sc(y_true, bmu_indices),
        "Grille": f"{grid_size[0]}x{grid_size[1]}"
    }

    if training_time is not None:
        results["Temps (s)"] = training_time

    return results

def create_results_dataframe_to_latex(results : list, round_digits : int = 3) -> pd.DataFrame:
    """
    Convertit une liste de résultats d'évaluation en un DataFrame pandas
    Args:
        results (list): Liste de dictionnaires contenant les résultats d'évaluation
        round_digits (int): Nombre de chiffres après la virgule pour l'arrondi des valeurs numériques
    Returns:
        pd.DataFrame: DataFrame contenant les résultats d'évaluation
    """
    df = pd.DataFrame(results)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(round_digits)

    df.to_latex("resultats_des_metriques.tex", index=False)
    
    return df

# =============================================================================================================================
# MATRICE U 
# =============================================================================================================================

def u_matrix_compute(poids: np.ndarray, grid_size: tuple) -> np.ndarray:
    """
    Calcule la U-Matrix pour visualiser les distances entre les prototypes du SOM
    Args:
        poids (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
        grid_size (tuple): Taille de la grille du SOM
    Returns:
        np.ndarray: U-Matrix de forme de grid_size
    """
    poids = flatten_weights(poids)

    r, c = grid_size

    if poids.shape[0] != r * c:
        raise ValueError(f"Incohérence entre grid_size={grid_size} et le nombre de poids={poids.shape[0]}.")


    grid_poids = poids.reshape(r, c, -1)
    
    u_matrix = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            neighbors = []

            if i > 0:  # Haut
                neighbors.append(grid_poids[i - 1, j])
            if i < r - 1:  # Bas
                neighbors.append(grid_poids[i + 1, j])
            if j > 0:  # Gauche
                neighbors.append(grid_poids[i, j - 1])
            if j < c - 1:  # Droite
                neighbors.append(grid_poids[i, j + 1])

            dists = [np.linalg.norm(grid_poids[i, j] - n) for n in neighbors]

            u_matrix[i, j] = np.mean(dists)

    return u_matrix

def plot_u_matrix(poids: np.ndarray, grid_size: tuple, title: str = "U-Matrix", save_path: str = None) -> None:
    """
    Affiche la U-Matrix en utilisant Matplotlib
    Args:
        u_matrix (np.ndarray): U-Matrix de forme (r, c)
        title (str): Titre du graphique
        save_path (str): Chemin pour sauvegarder l'image de la U-Matrix
    """
    u_matrix = u_matrix_compute(poids, grid_size)

    plt.figure(figsize=(6, 5))
    plt.imshow(u_matrix, cmap="viridis")
    plt.colorbar(label="Distance moyenne")
    plt.title(title)
    plt.xlabel("Colonne")
    plt.ylabel("Ligne")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight") 

    plt.show()


# =============================================================================================================================
# LABEL MAP
# =============================================================================================================================

def compute_label_maj_map(data: np.ndarray, poids: np.ndarray, y_true: np.ndarray, grid_size: tuple) -> np.ndarray:
    """
    Calcule une carte chaque noeud reçoit le label majoritaire
    des points qui lui sont associés.
    Args:
        data (np.ndarray): Données d'entrée de forme (n_samples, input_dim)
        poids (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
        y_true (np.ndarray): Labels vrais de forme (n_samples,)
        grid_size (tuple): Taille de la grille du SOM
    Returns:
        np.ndarray: Tableau de labels majoritaires pour chaque prototype du SOM
    """
    poids = flatten_weights(poids)
    
    bmu_indices, _ = compute_BMU(data, poids)
    r, c = grid_size

    if poids.shape[0] != r * c:
        raise ValueError(f"Incohérence entre grid_size={grid_size} et le nombre de poids={poids.shape[0]}.")

    label_map = np.full(r * c, fill_value=-1)

    for node in range(r * c):
        labels = y_true[bmu_indices == node]

        if len(labels) > 0:
            label_map[node] = mode(labels, keepdims=True).mode[0]

    return label_map.reshape(r, c)

def plot_label_map(data: np.ndarray, weights: np.ndarray, 
                   y_true: np.ndarray, grid_size: tuple, 
                   title: str = "Carte des labels", save_path: str = None)-> None:
    """
    Affiche la carte des labels majoritaires.
    Les noeuds sans point associé valent -1.
    Args:
        data (np.ndarray): Données d'entrée de forme (n_samples, input_dim)
        weights (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
        y_true (np.ndarray): Labels vrais de forme (n_samples,)
        grid_size (tuple): Taille de la grille du SOM
        title (str): Titre du graphique
        save_path (str): Chemin pour sauvegarder l'image de la carte des labels
    """
    label_map = compute_label_maj_map(data, weights, y_true, grid_size)

    plt.figure(figsize=(6, 5))
    plt.imshow(label_map, cmap="tab10")
    plt.colorbar(label="Label majoritaire")
    plt.title(title)
    plt.xlabel("Colonne")
    plt.ylabel("Ligne")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================================================================
# HIT MAP
# =============================================================================================================================

def hit_map(data: np.ndarray, poids: np.ndarray, grid_size: tuple, title: str = "Hit Map", save_path: str = None)-> None:
    """calcul la hit map du SOM, c'est à dire le nombre de points associés à chaque prototype du SOM, et l'affiche
    Args:        
        data (np.ndarray): Données d'entrée de forme (n_samples, input_dim)
        poids (np.ndarray): Codebook du SOM de forme (n_nodes, input_dim)
        grid_size (tuple): Taille de la grille du SOM
        title (str): Titre du graphique
        save_path (str): Chemin pour sauvegarder l'image de la hit map
    """
    poids = flatten_weights(poids)
    
    r, c = grid_size

    if poids.shape[0] != r * c:
        raise ValueError(f"Incohérence entre grid_size={grid_size} et le nombre de poids={poids.shape[0]}.")

    bmu_indices, _ = compute_BMU(data, poids)

    hit_map = np.zeros(r * c)

    for bmu in bmu_indices:
        hit_map[bmu] += 1

    
    hit_map = hit_map.reshape(r, c)

    plt.figure(figsize=(6, 5))
    plt.imshow(hit_map, cmap="magma")
    plt.colorbar(label="Nombre d'activations")
    plt.title(title)
    plt.xlabel("Colonne")
    plt.ylabel("Ligne")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================================================================
# COURBES D'APPRENTISSAGE
# =============================================================================================================================

def plot_training_curves(history: dict, title: str = "Courbes d'apprentissage", save_path: str = None)-> None:
    """
    Affiche plusieurs courbes d'apprentissage.

    Args:
        history: dict, par exemple :
            {
                "QE": [...],
                "Reconstruction loss": [...],
                "SOM loss": [...],
                "Total loss": [...]
            }
    """
    plt.figure(figsize=(7, 5))

    for name, values in history.items():
        plt.plot(values, label=name)

    plt.title(title)
    plt.xlabel("Époque / itération")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================================================================
# GRID SIZE STUDY
# =============================================================================================================================

def plot_grid_size_study(df: pd.DataFrame, metric: str, title: str = None, save_path: str = None) -> None:
    """
    Affiche l'évolution d'une métrique en fonction de la taille de grille.
    Args:
        df (pd.DataFrame): DataFrame contenant les résultats d'évaluation, avec au moins les colonnes "Modèle", "Grille" et la métrique à afficher
        metric (str): Nom de la colonne de la métrique à afficher
        title (str): Titre du graphique
        save_path (str): Chemin pour sauvegarder l'image du graphique
    """
    required_columns = {"Modèle", "Grille", "n_noeuds", metric}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans le DataFrame : {missing_columns}"
        )

    plt.figure(figsize=(7, 5))

    for model in df["Modèle"].unique():
        subset = df[df["Modèle"] == model].sort_values("n_noeuds")
        plt.plot(
            subset["n_noeuds"],
            subset[metric],
            marker="o",
            label=model
        )

    if title is None:
        title = f"Effet de la taille de grille sur {metric}"

    plt.title(title)
    plt.xlabel("Nombre de noeuds dans la grille")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================================================================
# CHARCHEMENT DE MNIST et FashionMNIST
# =============================================================================================================================


def load_dataset(name: str = "MNIST", root: str = None, 
                 normalize: bool = True, flatten: bool = True, 
                 n_train: int = None, n_test: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge MNIST ou Fashion-MNIST.

    Args:
        name: "MNIST" ou "FashionMNIST"
        root: dossier où télécharger les données
        normalize: si True, les pixels sont dans [0, 1]
        flatten: si True, transforme chaque image 28x28 en vecteur de taille 784
        n_train: nombre optionnel d'exemples d'entraînement
        n_test: nombre optionnel d'exemples de test

    Returns:
        X_train, y_train, X_test, y_test sous forme de numpy arrays
    """

    transform = transforms.ToTensor()

    if name == "MNIST":
        dataset_class = datasets.MNIST
    elif name == "FashionMNIST":
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError("name doit être 'MNIST' ou 'FashionMNIST'.")

    if root is None:
        root = "."

    train_dataset = dataset_class(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = dataset_class(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    if n_train is not None:
        train_dataset = Subset(train_dataset, range(n_train))

    if n_test is not None:
        test_dataset = Subset(test_dataset, range(n_test))

    X_train, y_train = dataset_to_numpy(train_dataset, flatten=flatten)
    X_test, y_test = dataset_to_numpy(test_dataset, flatten=flatten)

    if not normalize:
        X_train = X_train * 255.0
        X_test = X_test * 255.0

    return X_train, y_train, X_test, y_test


def dataset_to_numpy(dataset, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Convertit un dataset torchvision en tableaux numpy.
    Args:
        dataset: un dataset torchvision (ex: MNIST, FashionMNIST)
        flatten: si True, transforme chaque image 28x28 en vecteur de taille 784
    Returns:
        images, labels sous forme de numpy arrays
    """

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    images, labels = next(iter(loader))

    # images : (n_samples, 1, 28, 28)
    images = images.numpy()
    labels = labels.numpy()

    if flatten:
        images = images.reshape(images.shape[0], -1)
    else:
        images = images.squeeze(1)

    return images.astype(np.float32), labels.astype(np.int64)