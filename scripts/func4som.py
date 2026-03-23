import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def weight_t0(data: np.ndarray, grid_size: tuple, weight_init: str) -> np.ndarray:
    """
    Création du codebook du SOM selon le choix (random ou pca)

    Args:
        data (np.ndarray): Données d'entrée pour la création des poids
        grid_size (tuple): Taille de la grille du SOM
        weight_init (str): Méthode d'initialisation des poids

    Returns:
        np.ndarray: Codebook initialisé
    """

    if weight_init == 'random':
        # initialisation aléatoire des poids dans l'espace des données
        n_poids = grid_size[0] * grid_size[1]
        if n_poids > data.shape[0]:
            raise ValueError(f"Grid size {grid_size} ({n_poids} neurons) is too large for "
                            f"{data.shape[0]} samples when initializing without replacement.")
        indices = np.random.choice(len(data), n_poids, replace= False)
        return data[indices].reshape(grid_size[0], grid_size[1], data.shape[1])
        
    elif weight_init == 'pca':

        if data.shape[0] < 2:
            raise ValueError("PCA initialization requires dim(data) >= 2.")
        
        pca = PCA(n_components=2)
        pca.fit(data)

        mu = pca.mean_
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]

        codebook = np.zeros((grid_size[0], grid_size[1], data.shape[1]))
        x_range = np.linspace(-1, 1, grid_size[0])
        y_range = np.linspace(-1, 1, grid_size[1])

        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                codebook[i, j] = mu + x * pc1 + y * pc2

        return codebook

    else:
        raise ValueError(f"Unsupported weight initialization method: {weight_init}, choose 'random' or 'pca'")

def find_bmu(x: np.ndarray, codebook: np.ndarray, dis_metric) -> tuple[int, int]:
    """
    Calcul de l'index du best matching unit
    Args:
        x (np.ndarray): Vecteur pris du dataset
        codebook (np.ndarray): codebook actuel du SOM
        dis_metric: Fonction de distance

    Returns:
        bmu_idx(tuple[int, int]): index du poid gagnant
    """

    d = dis_metric(codebook,x) # calcul de la distance entre x et tous les poids du codebook

    bmu_idx = np.unravel_index(np.argmin(d), d.shape)    

    return bmu_idx

def neighborhood_func(bmuindx: tuple[int, int], t: int, rayon_init: float,
                      learning_rate_init: float, n_steps: int, grid_size: tuple[int,int]) -> np.ndarray:
        """"
        Fonction de voisinage pour la mise à jour des poids dans un rayon autours du BMU. Le rayon diminue au fil des itérations.
        Args:
            bmuindx (tuple[int, int]): Index du best matching unit
            t (int): Itération actuelle
            rayon_init (float): Rayon initial de voisinage
            learning_rate_init (float): Taux d'apprentissage initial
            n_steps (int): Nombre total d'étapes pour l'entraînement du SOM
            grid_size (tuple[int, int]): Taille de la grille du SOM
        Returns:
            np.ndarray: Matrice de voisinage pour la mise à jour des poids
        """
        
        # Mise à jour du rayon et du taux d'apprentissage
        rayon_t = rayon_init * np.exp(-t / n_steps)

        if learning_rate_init is not None:
            learning_rate_t = learning_rate_init * np.exp(-t / n_steps)
        else:
            learning_rate_t = 1

        grid_x, grid_y = np.indices(grid_size) # indices de la grille du SOM
        sqdist = (grid_x - bmuindx[0]) ** 2 + (grid_y - bmuindx[1]) ** 2
        
        # Calcul de la matrice de voisinage en fonction de la distance entre les neurones et le BMU
        hci = learning_rate_t * np.exp(-sqdist / (2 * rayon_t ** 2))
        return hci

def get_metrics(metric:str):
    """
    Arg:
        metric (str): Métrique de distance à utiliser ('euclidean', 'manhattan', 'cosine')
    Returns:
        function: Fonction de distance correspondante à la métrique choisie"""
    
    if metric == 'euclidean':
        return lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=-1))
    
    elif metric == 'manhattan':
        return lambda x, y: np.sum(np.abs(x - y), axis=-1)
    
    elif metric == 'cosine':
        return lambda x, y: 1 - np.sum(x * y, axis=-1) / np.maximum(np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-12)
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    

def plot_som_2D(codebook: np.ndarray, data: np.ndarray, y: np.ndarray, title: str = "SOM Visualization") -> None:
    """
    Visualisation du SOM en 2D
    Args:
        codebook (np.ndarray): Codebook du SOM à visualiser
        data (np.ndarray): Données d'entrée pour la visualisation
        y (np.ndarray): Étiquettes des données pour la coloration des points
        title (str): Titre du graphique
    """
    h, w, d = codebook.shape
    if d != 2:
        raise ValueError("Codebook must have 2 dimensions for 2D visualization.")

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:,0], data[:, 1], c=y, label='Data Points')
    plt.scatter(codebook[:, :, 0].flatten(), codebook[:, :, 1].flatten(), c='red', marker='X', label='SOM Neurons')

    for i in range(h):
        for j in range(w):
            if i+1 < h:
                plt.plot([codebook[i,j,0], codebook[i+1,j,0]],
                        [codebook[i,j,1], codebook[i+1,j,1]], 'k-', lw=1)
            if j+1 < w:
                plt.plot([codebook[i,j,0], codebook[i,j+1,0]],
                        [codebook[i,j,1], codebook[i,j+1,1]], 'k-', lw=1)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()