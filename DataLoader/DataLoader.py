# Import libraries
import os
from torch.utils.data import Dataset
import warnings
import trimesh
import glob
import numpy as np
import torch
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

# Dataset class
class QM9Dataset(Dataset):
    # Initialize the dataset object
    def __init__(self, data, find_edges, r_max=1):
        self.dataset = data
        self.find_edges = find_edges
        self.r_max = r_max

    # Return the length of the dataset
    def __len__(self):
        return len(self.dataset)
    
    # Find neighbours
    def find_neighbours(self, pos, r_max=1):
        # Get the positions of the atoms
        pos = pos.numpy()

        # Initialize the list for edges
        edge_index = []

        # Loop over all atoms
        for i, p_i in enumerate(pos):
            # Loop over all atoms again (including the same atom)
            for j, p_j in enumerate(pos):
                # Don't compare the same atom
                if i != j:
                    # Calculate distance
                    d = np.linalg.norm(p_i - p_j)

                    # Check if distance is below r_max
                    if d <= r_max:
                        edge_index.append([i, j])

        return edge_index
    
    # Return the item at the given index
    def __getitem__(self, index):
        batch = self.dataset[index]

        # Get the atomic numbers and coordinates
        atomic_numbers = batch.z
        coords = batch.pos

        # Find the edges
        if self.find_edges:
            edge_index = self.find_neighbours(coords, r_max=self.r_max)
            dataframe = np.hstack((atomic_numbers.reshape(-1, 1), coords))
            return dataframe, edge_index

        else:
            # Combine the atomic numbers and coordinates
            dataframe = np.hstack((atomic_numbers.reshape(-1, 1), coords))
            return dataframe