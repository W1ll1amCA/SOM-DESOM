import numpy as np
import scripts.utils_som as utils_som
import scripts.evaluation_utils as eval_utils

class stepwiseSOM:
    """
    Self-Organizing Map (SOM) en mode original (step-wise)
    """
    def __init__(self, 
                learning_rate_init: float, 
                rayon_init: float,
                grid_size: tuple, 
                n_steps: int,
                rayon_final: float = 0.1, 
                weight_init: str = 'random', 
                dis_choice: str = 'euclidean', 
                topology: str = 'rectangular'):
        
        """Initialisation du SOM en fonction de ses hyperparamètres et le type de SOM choisi (batch ou stepwise)

        Args:
            learning_rate_init (float): Taux d'apprentissage initial
            rayon_init (float): Rayon initial de voisinage
            rayon_final (float): Rayon final de voisinage
            grid_size (tuple): Taille de la grille du SOM
            n_steps (int): Nombre d'étapes pour l'entraînement du SOM
            weight_init (str): Méthode d'initialisation des poids random par défaut (ex: 'random', 'pca')
            dis_choice (str): Métrique de distance utilisée pour trouver le noeud gagnant euclidean par défaut (ex: 'euclidean', 'manhattan', etc.)
            topology (str): Topologie de la grille du SOM (ex: 'rectangular', 'hexagonal')

        """
        self.learning_rate_init = learning_rate_init
        self.grid_size = grid_size 
        self.n_steps = n_steps 
        self.weight_init = weight_init 
        self.dis_choice = dis_choice 
        self.topology = topology

        if rayon_init <= 0:
            raise ValueError("Rayon initial doit être positif.")
        self.rayon_init = rayon_init
        self.rayon_final = rayon_final

        self.dis_metric = utils_som.get_metrics(dis_choice) # fonction de dist sur la base de la métrique choisie
        
    def update_weights_stepwise(self, codebook: np.ndarray, x: np.ndarray, bmu_idx: tuple[int, ...], t: int) -> np.ndarray:
        """
        Mise à jour des poids du SOM en mode step-wise
        Args:
            codebook (np.ndarray): Codebook actuel du SOM
            x (np.ndarray): Vecteur d'entrée sélectionné aléatoirement du dataset
            bmu_idx (tuple[int, ...]): Index du best matching unit
            t (int): Itération actuelle
        Returns:
            np.ndarray: Codebook mis à jour
        """

        # fonction de voisinage
        hci = utils_som.neighborhood_func(bmu_idx, t, self.rayon_init, self.learning_rate_init, self.n_steps, self.grid_size, self.topology, self.rayon_final)

        # update du codebook
        codebook += hci[... , np.newaxis] * (x - codebook)

        return codebook
    
    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Entraînement du SOM en mode original (step-wise)
        Args:
            data (np.ndarray): Données d'entrée pour l'entraînement du SOM

        Returns:
            tuple: (codebook, history) où codebook est le codebook entraîné
                   et history contient la QE à chaque itération
        """
        history = {"QE": []}  # Historique des quantization errors pour chaque itération

        codebook = utils_som.weight_t0(data, self.grid_size, self.weight_init)

        for t in range(self.n_steps):
            
            x = data[np.random.randint(0, data.shape[0])] # sélection aléatoire d'un point du dataset

            # Trouver le poids gagant pour x
            bmu_idx = utils_som.find_bmu(x, codebook, self.dis_metric)

            # Mise à jour du poids gagnant et de ses voisins
            codebook = self.update_weights_stepwise(codebook, x, bmu_idx, t)

            # calcul du quantization error pour cette itération
            qe = eval_utils.err_quantification(data, codebook)
            history["QE"].append(qe)

        return codebook, history



class batchSOM:
    """
    Self-Organizing Map (SOM) en mode batch
    """

    def __init__(self, rayon_init: float, grid_size: tuple, n_steps: int, 
                  rayon_final: float = 0.1, weight_init: str = 'random', 
                  dis_choice: str = 'euclidean', topology: str = 'rectangular',
                  calculate_qe: bool = True):
        
        """Initialisation du SOM en fonction de ses hyperparamètres et le type de SOM choisi (batch ou stepwise)

        Args:
            rayon_init (float): Rayon initial de voisinage
            rayon_final (float): Rayon final de voisinage
            grid_size (tuple): Taille de la grille du SOM
            n_steps (int): Nombre d'étapes pour l'entraînement du SOM
            weight_init (str): Méthode d'initialisation des poids random par défaut (ex: 'random', 'pca')
            dis_choice (str): Métrique de distance utilisée pour trouver le noeud gagnant euclidean par défaut (ex: 'euclidean', 'manhattan', etc.)
            topology (str): Topologie de la grille du SOM (ex: 'rectangular', 'hexagonal')
            calculate_qe (bool): Indique si le quantization error doit être calculé à chaque itération (True par défaut)
        """
        self.grid_size = grid_size 
        self.n_steps = n_steps 
        self.weight_init = weight_init 
        self.dis_choice = dis_choice 
        self.topology = topology
        self.calculate_qe = calculate_qe

        if rayon_init <= 0:
            raise ValueError("Rayon initial doit être positif.")
        self.rayon_init = rayon_init
        self.rayon_final = rayon_final
        self.dis_metric = utils_som.get_metrics(dis_choice) # fonction de dist sur la base de la métrique choisie


    def train(self, data: np.ndarray) -> np.ndarray:
        """
        Entraînement du SOM en mode batch
        Args:
            data (np.ndarray): Données d'entrée pour l'entraînement du SOM

        Returns:
            tuple: (codebook, history) où codebook est le codebook entraîné
                   et history contient la QE à chaque itération
        """

        history = {"QE": []}  # Historique des quantization errors pour chaque itération

        codebook = utils_som.weight_t0(data, self.grid_size, self.weight_init)
    
        for t in range(self.n_steps):                    
            nodes = {}

            # remplissage de liste des points associés à chaque noeud
            for x in data:
                bmu_idx = utils_som.find_bmu(x, codebook, self.dis_metric)
                if bmu_idx not in nodes:
                    nodes[bmu_idx] = []
                nodes[bmu_idx].append(x)    

            # mise à jour du codebook par la moyenne des points associés à chaque noeud
            codebook = self.update_weights_batch(codebook, nodes, t)

            # calcul du quantization error pour cette itération
            if self.calculate_qe:
                qe = eval_utils.err_quantification(data, codebook)
                history["QE"].append(qe)

        return codebook, history
   
    
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
            
            hci = utils_som.neighborhood_func(bmu_idx, t, self.rayon_init, None, self.n_steps, self.grid_size, self.topology, self.rayon_final)

            numerator += hci[..., np.newaxis] * mean * size
            denominator += hci * size
        
        mask = denominator > 0
        new_codebook = np.copy(codebook)
        new_codebook[mask] = numerator[mask] / denominator[mask][..., np.newaxis]
        
        return new_codebook
