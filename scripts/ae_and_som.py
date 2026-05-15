import torch
import torch.nn as nn
import torch.optim as optim
from scripts.som import batchSOM
from torch.utils.data import TensorDataset, DataLoader
from scripts.utils_desom import Autoencoder

class AESOM:
    """Classe combinant un autoencodeur et un SOM"""

    def __init__(self, input_dim: int, 
                 latent_dim: int, 
                 rayon_init: float, 
                 grid_size: tuple, 
                 n_steps_som: int, 
                 weight_init: str = 'random', 
                 dis_choice: str = 'euclidean', 
                 topology: str = 'rectangular',
                 device: str = None):
        
        """Initialisation de la classe combinée
        Args:
            input_dim (int): Dimension de l'entrée
            latent_dim (int): Dimension de l'espace latent
            rayon_init (float): Rayon initial pour le SOM
            grid_size (tuple): Taille de la grille pour le SOM
            n_steps_som (int): Nombre d'étapes pour l'entraînement du SOM
            weight_init (str): Méthode d'initialisation des poids pour le SOM (ex: 'random', 'pca')
            dis_choice (str): Choix de la distance pour le SOM (ex: 'euclidean', 'manhattan', etc.)
            topology (str): Topologie pour le SOM (ex: 'rectangular', 'hexagonal')
            device (str): Périphérique pour l'entraînement (ex: 'cpu', 'cuda')
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialisation de l'autoencodeur et du SOM
        self.autoencoder = Autoencoder(input_dim, latent_dim).to(self.device)
        self.som = batchSOM(rayon_init, grid_size, n_steps=n_steps_som, rayon_final=0.1, 
                            weight_init=weight_init, dis_choice=dis_choice, topology=topology,
                            calculate_qe=False)

    def train_ae(self, data, n_epochs=100, batch_size=32, lr=1e-3):
        """
        Entraîne seulement l'autoencodeur.
        Args:
            data (torch.Tensor): Données d'entrée pour l'entraînement de l'autoencodeur
            n_epochs (int): Nombre d'époques pour l'entraînement
            batch_size (int): Taille des lots pour l'entraînement
            lr (float): Taux d'apprentissage pour l'optimiseur
        returns:
            losses (list): Liste des pertes d'entraînement par époque
        """

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.float().to(self.device)

        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.autoencoder.train()
        losses = []

        for epoch in range(n_epochs):

            epoch_loss = 0.0

            for batch in dataloader:
                x = batch[0]

                optimizer.zero_grad()

                x_hat, z = self.autoencoder(x)

                loss_reconstruction = criterion(x_hat, x)

                loss_reconstruction.backward()
                optimizer.step()

                epoch_loss += loss_reconstruction.item() * x.size(0)

            epoch_loss /= len(dataset)
            losses.append(epoch_loss)

        return losses

    def encode_latent(self, data, batch_size=256):
        """
        Encode toutes les données dans l'espace latent.
        Args:
            data (torch.Tensor): Données d'entrée à encoder
            batch_size (int): Taille des lots pour l'encodage
        returns:
                np.ndarray: Représentations latentes de toutes les données
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.float().to(self.device)

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.autoencoder.eval()

        z_all = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                _, z = self.autoencoder(x)
                z_all.append(z.cpu())

        return torch.cat(z_all, dim=0).numpy()
    
    def train_som(self, data):
        """
        Entraîne le SOM sur toutes les représentations latentes.
        Args:
            data (torch.Tensor): Données d'entrée à encoder et à utiliser pour l'entraînement du SOM
        returns:
            np.ndarray: Codebook entraîné du SOM
        """
        z_np = self.encode_latent(data)
        return self.som.train(z_np)
    
    def train_ae_n_som(self, data, n_epochs=100, batch_size=32, lr=1e-3):
        """
        Entraînement complet en deux étapes :
        1. autoencodeur
        2. SOM sur l'espace latent
        Args:
            data (torch.Tensor): Données d'entrée pour l'entraînement
            n_epochs (int): Nombre d'époques pour l'entraînement de l'autoencodeur
            batch_size (int): Taille des lots pour l'entraînement de l'autoencodeur
            lr (float): Taux d'apprentissage pour l'optimiseur de l'autoencodeur
        returns:
            tuple: (ae_losses, som_result) où ae_losses est la liste des pertes d'entraînement de l'autoencodeur et som_result est le codebook entraîné du SOM
        """
        ae_losses = self.train_ae(data, n_epochs, batch_size, lr)
        som_result = self.train_som(data)

        return ae_losses, som_result