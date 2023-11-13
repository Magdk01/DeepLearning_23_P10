# Import libraries
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader

# Dataset class
class QM9Dataset(Dataset):
    # Initialize the dataset object
    def __init__(self, data, y_index=0):
        self.dataset = data
        self.y_index = y_index

    # Return the length of the dataset
    def __len__(self):
        return len(self.dataset)
    
    # Return the item at the given index
    def __getitem__(self, index):
        batch = self.dataset[index]

        # Get the atomic numbers and coordinates
        atomic_numbers = batch.z
        coords = batch.pos

        # q = atomic_numbers copied 3 times 
        q = [num.item() for num in atomic_numbers for _ in range(3)]

        # R_ij = 3x(len(atomic_numbers) * 3))
        r_ij = torch.zeros((3, len(atomic_numbers) * 3))

        # Get the distance between atoms
        for i in range(r_ij.shape[0]):
            for j in range(r_ij.shape[1]):
                r_ij[i][j] = coords[j//3][i] - coords[j%3][i]

        # Variable to predict
        y = batch.y[0][self.y_index]

        # Combine the atomic numbers, coordinates and variable to predict
        return q, r_ij, y

def my_collate_fn(data):
        return data

# Function to create the dataset
def DataLoad(batch_size=1, shuffle=False, split=[0.8, 0.1, 0.1], y_index=0):
    # Create the dataset
    data = QM9(root='./QM9')

    # Create the dataset object
    dataset = QM9Dataset(data, y_index=0)

    # Train-test split
    train_size = int(split[0] * len(dataset))
    test_size = int(split[1] * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    # Create the train and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn)

    return train_dataloader, test_dataloader, val_dataloader