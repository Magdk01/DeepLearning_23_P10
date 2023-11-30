# Import libraries
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader


# Dataset class
class QM9Dataset(Dataset):
    # Initialize the dataset object
    def __init__(self, data, target_index):
        self.dataset = data
        self.target_index = target_index

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
        y = batch.y[0][self.target_index]

        # Combine the atomic numbers, coordinates and variable to predict
        return atomic_numbers, coords, y


def my_collate_fn(batch):
    modified_batch = []
    indexes = []

    for index, data in enumerate(batch):
        data = list(data)
        num_atoms = data[1].shape[0]  # Get the number of atoms
        indexes += [
            int(index)
        ] * num_atoms  # Repeat the index for each atom in the molecule

        # No modification to the atoms' coordinates
        modified_batch.append(data)

    # Return the batch data and the indexes
    return modified_batch, torch.tensor(indexes)


# Function to create the dataset
def DataLoad(batch_size=1, shuffle=False, split=[0.8, 0.1, 0.1], target_index=0):
    # Create the dataset
    data = QM9(root="./QM9")

    # Create the dataset object
    dataset = QM9Dataset(data, target_index)

    total_size = len(dataset) - 64  # Reserve 64 for the validation set
    train_size = int(split[0] * total_size)
    test_size = total_size - train_size  # Adjust test size accordingly

    # Validation size is set to 64
    val_size = 64

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, val_size]
    )

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate_fn
    )

    return train_dataloader, test_dataloader, val_dataloader