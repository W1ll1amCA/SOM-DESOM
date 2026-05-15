import numpy as np
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

    data = np.asarray(data, dtype=float)
    n_samples, input_dim = data.shape
    n_poids = int(np.prod(grid_size))
    
    # initialisation aléatoire des poids dans l'espace des données
    if weight_init == 'random':
    
        if n_poids > n_samples:
            raise ValueError("Grid size is too large for the number of data points. Reduce grid size.")
        
        indices = np.random.choice(n_samples, n_poids, replace= False)
        return data[indices].reshape(*grid_size, input_dim)
    
    # initialisation des poids par PCA
    elif weight_init == 'pca':
       
        if n_samples < 2:
            raise ValueError("PCA initialization requires at least 2 samples.")

        map_dim = len(grid_size)
 
        n_components = min(map_dim, input_dim, n_samples)

        pca = PCA(n_components=n_components)
        pca.fit(data)

        mu = pca.mean_
        comps = pca.components_

        grids = np.meshgrid(*[np.linspace(-1, 1, size) for size in grid_size], indexing="ij")
        coords = np.stack([g.ravel() for g in grids], axis=1)
        flat_codebook = np.tile(mu, (n_poids, 1))

        for k in range(n_components):
            flat_codebook += coords[:, k:k+1] * comps[k]

        return flat_codebook.reshape(*grid_size, input_dim)

    else:
        raise ValueError(f"Unsupported weight initialization method: {weight_init}, choose 'random' or 'pca'")

def find_bmu(x: np.ndarray, codebook: np.ndarray, dis_metric) -> tuple[int, ...]:
    """
    Calcul de l'index du best matching unit
    Args:
        x (np.ndarray): Vecteur pris du dataset
        codebook (np.ndarray): codebook actuel du SOM
        dis_metric: Fonction de distance

    Returns:
        bmu_idx(tuple[int, ...]): index du poid gagnant
    """

    d = dis_metric(codebook,x) # calcul de la distance entre x et tous les poids du codebook

    bmu_idx = np.unravel_index(np.argmin(d), d.shape)    

    return bmu_idx

def neighborhood_func(bmu_idx: tuple[int, ...], 
                      t: int, rayon_init: float,
                      learning_rate_init: float, 
                      n_steps: int, 
                      grid_size: tuple[int, ...], 
                      topology: str,
                      rayon_final: float) -> np.ndarray:
        """"
        Fonction de voisinage pour la mise à jour des poids dans un rayon autours du BMU. Le rayon diminue au fil des itérations.
        Args:
            bmu_indx (tuple[int, ...]): Index du best matching unit
            t (int): Itération actuelle
            rayon_init (float): Rayon initial de voisinage
            learning_rate_init (float): Taux d'apprentissage initial
            n_steps (int): Nombre total d'étapes pour l'entraînement du SOM
            grid_size (tuple[int, ...]): Taille de la grille du SOM
            topology (str): Topologie de la grille du SOM (ex: 'rectangular', 'hexagonal')
        Returns:
            np.ndarray: Matrice de voisinage pour la mise à jour des poids
        """
        if rayon_init <= 0:
                raise ValueError("rayon_init doit être positif.")

        if n_steps <= 0:
            raise ValueError("n_steps doit être positif.")

        # Mise à jour du rayon et du taux d'apprentissage
        rayon_t = rayon_init * (rayon_final / rayon_init) ** (t / (n_steps - 1))

        if learning_rate_init is not None:
            learning_rate_t = learning_rate_init * np.exp(-t / n_steps)
        else:
            learning_rate_t = 1

        # distances entre les noeuds et le BMU selon la topologie choisie
        sqdist = map_dist(grid_size, bmu_idx, topology)
        
        # Calcul de la matrice de voisinage en fonction de la distance entre les noeuds et le BMU
        hci = learning_rate_t * np.exp(-(sqdist**2) / (2 * rayon_t ** 2))

        return hci

def map_dist(grid_size: tuple[int, ...], bmu_idx: tuple[int, ...], topology: str) -> np.ndarray:
    """
    Calcul de la distance entre les noeuds et le BMU selon la topologie choisie
    Args:
        grid_size (tuple[int, ...]): Taille de la grille du SOM
        bmu_idx (tuple[int, ...]): Index du best matching unit
        topology (str): Topologie de la grille du SOM (ex: 'rectangular', 'hexagonal', 'toroidal')
    Returns:
        np.ndarray: Matrice des distances entre les noeuds et le BMU selon la topologie choisie
    """
    if topology == 'rectangular':
        return rectangular_dist(grid_size, bmu_idx)
    
    elif topology == 'hexagonal':
        return hexagonal_dist(grid_size, bmu_idx)
    
    elif topology == 'toroidal':
        return toroidal_dist(grid_size, bmu_idx)
    else:
        raise ValueError(f"Unsupported topology: {topology}, choose 'rectangular', 'toroidal' or 'hexagonal'")


def rectangular_dist(grid_size: tuple[int, ...], bmu_idx: tuple[int, ...]) -> np.ndarray:
        """
        calcul de la distance euclidienne entre les noeuds et le BMU pour une topologie rectangulaire
        Args:
            grid_size (tuple[int, ...]): Taille de la grille du SOM
            bmu_idx (tuple[int, ...]): Index du best matching unit
        Returns:
            np.ndarray: Matrice des distances euclidiennes entre les noeuds et le BMU
        """
        grid = np.indices(grid_size)
        
        bmu_array = np.array(bmu_idx).reshape((len(grid_size),) + (1,) * len(grid_size))

        return np.sqrt(np.sum((grid - bmu_array) ** 2, axis=0))


def toroidal_dist(grid_size: tuple[int, ...], bmu_idx: tuple[int, ...]) -> np.ndarray:
    """
    calcul de la distance euclidienne entre les noeuds et le BMU pour une topologie toroidale
    Args:
        grid_size (tuple[int, ...]): Taille de la grille du SOM
        bmu_idx (tuple[int, ...]): Index du best matching unit
    Returns:
        np.ndarray: Matrice des distances euclidiennes entre les noeuds et le BMU pour une topologie toroidale
    """
    grid = np.indices(grid_size)
    bmu_array = np.array(bmu_idx).reshape((len(grid_size),) + (1,) * len(grid_size))

    diff = np.abs(grid - bmu_array)

    for axis, size in enumerate(grid_size):
        diff[axis] = np.minimum(diff[axis], size - diff[axis])

    return np.sqrt(np.sum(diff ** 2, axis=0))    



def hexagonal_dist(grid_size: tuple[int, ...], bmu_idx: tuple[int, ...]) -> np.ndarray:
    """
    calcul de la distance hexagonale entre les noeuds et le BMU pour une topologie hexagonale
    seulement pour une grille 2D
    Args:
        grid_size (tuple[int, ...]): Taille de la grille du SOM
        bmu_idx (tuple[int, ...]): Index du best matching unit
    Returns:
        np.ndarray: Matrice des distances hexagonales entre les noeuds et le BMU pour une topologie hexagonale
    """

    if len(grid_size) != 2:
        raise ValueError("La topologie hexagonale est définie seulement pour une grille 2D.")

    r, c = grid_size
    r0, q0 = bmu_idx

    rr, qq = np.indices((r, c))

    # Coordonnées axiales approximatives pour grille hexagonale en offset
    x1 = qq - (rr // 2)
    z1 = rr
    y1 = -x1 - z1

    x0 = q0 - (r0 // 2)
    z0 = r0
    y0 = -x0 - z0

    dist = (np.abs(x1 - x0) + np.abs(y1 - y0)+ np.abs(z1 - z0)) / 2

    return dist


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
        return lambda x, y: 1 - (np.sum(y * x, axis=-1) / np.maximum(np.linalg.norm(y, axis=-1) * np.linalg.norm(x), 1e-12))
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    