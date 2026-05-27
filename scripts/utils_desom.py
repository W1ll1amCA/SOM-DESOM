import torch
import torch.nn as nn
import itertools


class Encodeur(nn.Module):
    """Encodeur pour l'autoencodeur"""
    def __init__(self, input_dim : int, latent_dim : int):
        super().__init__()
        # Construction d'un réseau de neurones simple pour l'encodeur
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, latent_dim) 
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Passage de l'entrée à travers l'encodeur
        Args:
            x (torch.Tensor): Entrée de dimension (batch_size, input_dim)
        Returns:
            torch.Tensor: Représentation latente de dimension (batch_size, latent_dim)
        """
        return self.net(x)

class Decodeur(nn.Module):
    """Decodeur pour l'autoencodeur"""

    def __init__(self, latent_dim : int, output_dim : int):
        """Initialisation du decodeur
        Args:
            latent_dim (int): Dimension de l'espace latent
            output_dim (int): Dimension de la sortie (même que l'entrée)
        """
        super().__init__()
        # Construction d'un réseau de neurones simple pour le decodeur
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z : torch.Tensor) -> torch.Tensor:
        """Passage de la représentation latente à travers le decodeur
        Args:
            z (torch.Tensor): Représentation latente de dimension (batch_size, latent_dim)
        Returns:
            torch.Tensor: Sortie de dimension (batch_size, output_dim)
        """
        return self.net(z)

class Autoencoder(nn.Module):
    """Autoencodeur complet avec encodeur et decodeur"""

    def __init__(self, input_dim : int, latent_dim : int):
        """Initialisation de l'autoencodeur
        Args:
            input_dim (int): Dimension de l'entrée
            latent_dim (int): Dimension de l'espace latent
        """
        super().__init__()
        # Initialisation de l'encodeur et du decodeur
        self.encodeur = Encodeur(input_dim, latent_dim)
        self.decodeur = Decodeur(latent_dim, input_dim)

    def forward(self, x : torch.Tensor)-> tuple:
        """Passage de l'entrée à travers l'autoencodeur
        Args:
            x (torch.Tensor): Entrée de dimension (batch_size, input_dim)
        Returns:
            tuple: (x_hat, z) où x_hat est la reconstruction de l'entrée et z est la représentation latente
        """

        # creation de l'espace latent
        z = self.encodeur(x)

        # reconstruction de l'entrée à partir de l'espace latent
        x_hat = self.decodeur(z)

        return x_hat, z
    

class SOMtorch(nn.Module):

    """SOM implémenté en PyTorch pour être utilisé dans un autoencodeur"""

    def __init__(self, grid_size : tuple, latent_dim : int):
        """Initialisation du SOM
        Args:
            grid_size (tuple): Taille de la grille du SOM 
            latent_dim (int): Dimension de l'espace latent
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.n_nodes = int(torch.prod(torch.tensor(grid_size)).item())
        self.latent_dim = latent_dim

        # Initialisation des poids du SOM de manière aléatoire
        self.weights = nn.Parameter(torch.rand(self.n_nodes, latent_dim))
        
        # Création d'une liste de coordonnées pour chaque nœud du SOM
        coords = list(itertools.product(*[range(s) for s in grid_size]))
        
        self.register_buffer('coords', torch.tensor(coords, dtype=torch.float))

    def forward(self, z : torch.Tensor, sigma : float) -> torch.Tensor:
        """Calcul du SOM pour une entrée donnée
        Args:
            z (torch.Tensor): Représentation latente de dimension (batch_size, latent_dim)
            sigma (float): Largeur de la fonction de voisinage
        Returns:
            tuple: (loss_SOM, bmu_indices) où loss_SOM est la perte du SOM et bmu_indices sont les indices des BMU
        """
        dist = torch.cdist(z, self.weights)**2  # Calcul des distances entre z et les poids du SOM
        
        bmu_indices = torch.argmin(dist, dim=1)  # Indices des Best Matching Units (BMU) pour chaque échantillon

        bmu_coords = self.coords[bmu_indices]  # Coordonnées des BMU dans la grille du SOM

        grid_dist = torch.cdist(bmu_coords, self.coords)** 2  # Distances au carré entre le BMU c_i et tous les noeuds

        h = torch.exp(-grid_dist / (2 * sigma**2))  # Fonction de voisinage gaussienne

        h = h.detach()  # Détacher h du graphe de calcul pour éviter les gradients à travers h

        loss_SOM = ((h * dist).sum(dim=1) / h.sum(dim=1).clamp_min(1e-12)).mean()   # Perte du SOM

        return loss_SOM, bmu_indices


