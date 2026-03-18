import numpy as np
import scripts.func4som as func4som

class stepwiseSOM:
    """
    Self-Organizing Map (SOM) en mode original (step-wise)
    """
    def __init__(self, learning_rate_init: float, rayon_init: float, grid_size: tuple, n_steps: int, weight_init: str, dis_choice: str):
        """Initialisation du SOM en fonction de ses hyperparamètres et le type de SOM choisi (batch ou stepwise)

        Args:
            learning_rate_init (float): Taux d'apprentissage initial
            rayon_init (float): Rayon initial de voisinage
            grid_size (tuple): Taille de la grille du SOM
            n_steps (int): Nombre d'étapes pour l'entraînement du SOM
            weight_init (str): Méthode d'initialisation des poids (ex: 'random', 'pca')
            dis_choice (str): Métrique de distance utilisée pour trouver le neurone gagnant (ex: 'euclidean', 'manhattan', etc.)
        """
        self.learning_rate_init = learning_rate_init
        self.grid_size = grid_size 
        self.n_steps = n_steps 
        self.weight_init = weight_init 
        self.dis_choice = dis_choice 

        if rayon_init <= 0:
            raise ValueError("Rayon initial doit être positif.")
        self.rayon_init = rayon_init
        
        self.dis_metric = func4som.get_metrics(dis_choice) # fonction de dist sur la base de la métrique choisie
        
    def update_weights_stepwise(self, codebook: np.ndarray, x: np.ndarray, bmu_idx: tuple[int, int], t: int) -> np.ndarray:
        """
        Mise à jour des poids du SOM en mode step-wise
        Args:
            codebook (np.ndarray): Codebook actuel du SOM
            x (np.ndarray): Vecteur d'entrée sélectionné aléatoirement du dataset
            bmu_idx (tuple[int, int]): Index du best matching unit
            t (int): Itération actuelle
        Returns:
            np.ndarray: Codebook mis à jour
        """

        # fonction de voisinage
        hci = func4som.neighborhood_func(bmu_idx, t, self.rayon_init, self.learning_rate_init, self.n_steps, self.grid_size)

        codebook += hci[:, :, np.newaxis] * (x - codebook)

        return codebook
    
    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Entraînement du SOM en mode original (step-wise)
        Args:
            data (np.ndarray): Données d'entrée pour l'entraînement du SOM

        Returns:
            np.ndarray: Codebook entraîné
        """
        codebook = func4som.weight_t0(data, self.grid_size, self.weight_init)

        for t in range(self.n_steps):
            
            x = data[np.random.randint(0, data.shape[0])] # sélection aléatoire d'un point du dataset

            # Trouver le poids gagant pour x
            bmu_idx = func4som.find_bmu(x, codebook, self.dis_metric)

            # Mise à jour du poids gagnant et de ses voisins
            codebook = self.update_weights_stepwise(codebook, x, bmu_idx, t)

        return codebook



class batchSOM:
    """
    Self-Organizing Map (SOM) en mode batch
    """

    def __init__(self, rayon_init: float, grid_size: tuple, n_steps: int, weight_init: str, dis_choice: str):
        """Initialisation du SOM en fonction de ses hyperparamètres et le type de SOM choisi (batch ou stepwise)

        Args:
            learning_rate_init (float): Taux d'apprentissage initial
            rayon_init (float): Rayon initial de voisinage
            grid_size (tuple): Taille de la grille du SOM
            n_steps (int): Nombre d'étapes pour l'entraînement du SOM
            weight_init (str): Méthode d'initialisation des poids (ex: 'random', 'pca')
            dis_choice (str): Métrique de distance utilisée pour trouver le neurone gagnant (ex: 'euclidean', 'manhattan', etc.)
        """
        self.grid_size = grid_size 
        self.n_steps = n_steps 
        self.weight_init = weight_init 
        self.dis_choice = dis_choice 

        if rayon_init <= 0:
            raise ValueError("Rayon initial doit être positif.")
        self.rayon_init = rayon_init
        
        self.dis_metric = func4som.get_metrics(dis_choice) # fonction de dist sur la base de la métrique choisie


    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Entraînement du SOM en mode batch
        Args:
            data (np.ndarray): Données d'entrée pour l'entraînement du SOM

        Returns:
            np.ndarray: Codebook entraîné
        """

        codebook = func4som.weight_t0(data, self.grid_size, self.weight_init)
    
        for t in range(self.n_steps):                    
            nodes = {}

            # remplissage de liste des points associés à chaque noeud
            for x in data:
                bmu_idx = func4som.find_bmu(x, codebook, self.dis_metric)
                if bmu_idx not in nodes:
                    nodes[bmu_idx] = []
                nodes[bmu_idx].append(x)    

            # mise à jour du codebook par la moyenne des points associés à chaque noeud
            codebook = self.update_weights_batch(codebook, nodes, t, self.rayon_init, None, self.n_steps, self.grid_size)

        return codebook
   
    
    def update_weights_batch(self, codebook: np.ndarray, nodes: dict, t: int) -> np.ndarray:
        """
        Mise à jour des poids du SOM en mode batch
        Args:
            codebook (np.ndarray): Codebook actuel du SOM
            nodes (dict): Dictionnaire associant chaque noeud à la liste des points qui lui sont associés
            t (int): Itération actuelle
        Returns:
            np.ndarray: Codebook mis à jour
        """

        group_means = {bmu_idx: np.mean(points, axis=0) for bmu_idx, points in nodes.items()}
        group_sizes = {bmu_idx: len(points) for bmu_idx, points in nodes.items()}
        numerator = np.zeros_like(codebook)
        denominator = np.zeros(self.grid_size)

        for bmu_idx, mean in group_means.items():
            size = group_sizes[bmu_idx]
            hci = func4som.neighborhood_func(bmu_idx, t, self.rayon_init, None, self.n_steps, self.grid_size)

            numerator += hci[:, :, np.newaxis] * mean * size
            denominator += hci * size
        
        denominator = np.where(denominator == 0, 1, denominator)  # éviter la division par zéro

        new_codebook = numerator / denominator[:, :, np.newaxis]
        return new_codebook

