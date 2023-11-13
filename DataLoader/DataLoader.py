# Import libraries
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader

# Dataset class
class QM9Dataset(Dataset):
    # Initialize the dataset object
    def __init__(self, data):
        self.dataset = data

    # Return the length of the dataset
    def __len__(self):
        return len(self.dataset)
    
    # Return the item at the given index
    def __getitem__(self, index):
        batch = self.dataset[index]

        # Get the atomic numbers and coordinates
        atomic_numbers = batch.z
        coords = batch.pos

        # Variable to predict
        y = batch.y

        # Combine the atomic numbers, coordinates and variable to predict
        return atomic_numbers, coords, y

def my_collate_fn(data):
        return data

# Function to create the dataset
def DataLoad(batch_size=1, shuffle=False, split=0.8):
    # Create the dataset
    data = QM9(root='./QM9')

    # Create the dataset object
    dataset = QM9Dataset(data)

    # Train-test split
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create the train and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn)

    return train_dataloader, test_dataloader