import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scripts.utils_desom as utils_desom


class DESOM(nn.Module):
    """Deep Embedded Self-Organizing Map (DESOM)"""

    def __init__(self, grid_size : tuple, input_dim : int, latent_dim : int, gamma: float = 1.0, device: str = None):
        """Initialisation du DESOM
        Args:
            grid_size (tuple): Taille de la grille du SOM
            input_dim (int): Dimension de l'entrée
            latent_dim (int): Dimension de l'espace latent
            gamma (float): Poids de la perte du SOM dans la perte totale
        """
        super().__init__()

        self.grid_size = grid_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        self.som = utils_desom.SOMtorch(grid_size, latent_dim)
        self.ae = utils_desom.Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
      
        self.to(self.device)

    def forward(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Passage de l'entrée à travers le DESOM
        Args:
            x (torch.Tensor): Entrée de dimension (batch_size, input_dim)
            sigma (float): Largeur de la fonction de voisinage pour le SOM
        Returns:
            torch.Tensor: Perte totale du DESOM
        """
        x_hat, z = self.ae(x)  # Passage de l'entrée à travers l'autoencodeur

        loss_som, bmu_indices = self.som(z, sigma) # Passage de la représentation latente à travers le SOM pour calculer la perte et les indices des BMU

        return x_hat, z, loss_som, bmu_indices

    def embedded_train(self, data, n_epochs=100, batch_size=32, lr=1e-3, sigma_init=1.0, sigma_final=0.1):
        """Entraîne le DESOM conjointement
        Args:
            data (torch.Tensor): Données d'entrée pour l'entraînement du DESOM
            n_epochs (int): Nombre d'époques pour l'entraînement
            batch_size (int): Taille des lots pour l'entraînement
            lr (float): Taux d'apprentissage pour l'optimiseur
            sigma_init (float): Largeur initiale de la fonction de voisinage pour le SOM
            sigma_final (float): Largeur finale de la fonction de voisinage pour le SOM
        returns:
            losses (list): Liste des pertes totales d'entraînement par époque
        """

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.float().to(self.device)

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = {
        "total": [],
        "reconstruction": [],
        "som": []}

        self.train()

        for epoch in range(n_epochs):

            epoch_loss = 0.0 # Perte totale
            epoch_rec = 0.0 # Reconstruction loss
            epoch_som = 0.0 # SOM loss

            # Calcul du sigma pour cette époque
            if n_epochs == 1:
                sigma = sigma_final
            else:
                sigma = sigma_init * (sigma_final / sigma_init) ** (epoch / (n_epochs - 1))

            for batch in dataloader:
                x = batch[0]

                optimizer.zero_grad()

                x_hat, z, loss_som, bmu_indices = self.forward(x, sigma)
                loss_rec = criterion(x_hat, x)

                # Perte totale introduisant le SOM avec un gamma
                loss_total = loss_rec + self.gamma * loss_som

                loss_total.backward()
                optimizer.step()

                batch_size_actual = x.size(0)

                epoch_loss += loss_total.item() * batch_size_actual
                epoch_rec += loss_rec.item() * batch_size_actual
                epoch_som += loss_som.item() * batch_size_actual

            n = len(dataset)

            losses["total"].append(epoch_loss / n)
            losses["reconstruction"].append(epoch_rec / n)
            losses["som"].append(epoch_som / n)

        return losses
    
    def encode(self, data, batch_size: int = 256):
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

        self.eval()

        z_all = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                _, z = self.ae(x)
                z_all.append(z.cpu())

        return torch.cat(z_all, dim=0)

    def predict_bmu(self, data, batch_size: int = 256):
        """
        Retourne le BMU de chaque donnée.
        Args:
            data (torch.Tensor): Données d'entrée pour lesquelles prédire les BMU
            batch_size (int): Taille des lots pour la prédiction
        returns:
            np.ndarray: Indices des BMU pour chaque donnée
        """

        z = self.encode(data, batch_size=batch_size).to(self.device)

        self.eval()

        with torch.no_grad():
            dist = torch.cdist(z, self.som.weights) ** 2
            bmu_indices = torch.argmin(dist, dim=1)

        return bmu_indices.cpu()

