from painn import painn
from DataLoader.DataLoader import DataLoad
import torch
from tqdm import tqdm
import numpy as np

import wandb

# WandB setup
wandb.login()
wandb.init(project="painn",
    config={
"learning_rate": 0.001,
"epochs": 1,
"batch_size": 1,
})


def run_epoch(loader, model, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.
    
    for i, batch in enumerate(loader):
        for j, x in enumerate(batch):
            atomic_numbers, coords, y = x

            atomic_numbers = np.array(atomic_numbers)
            coords = np.array(coords)
            
            y = y[0][0]

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(atomic_numbers, coords)
            assert not np.isnan(outputs.item()) 

            # Compute the loss and its gradients
            #loss = loss_fn(float(outputs), float(y))
            # Squared error as loss function
            loss = (outputs - y)**2
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()
            
            
            running_loss += loss.item()
            
            if j == len(batch) - 1:
                last_loss = running_loss / len(batch) # loss per batch
                print('batch {} loss: {} n atoms {}'.format(i + 1, last_loss, len(atomic_numbers)))
                running_loss = 0.
                wandb.log({"Loss": last_loss})
    return last_loss

def main():
    train_loader, test_loader, val_loader = DataLoad()
    model = painn()
    loss = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 1
    
    for i in tqdm(range(EPOCHS)):
        print("Training")
        train_loss = run_epoch(train_loader, model, loss, optimizer)
        print(train_loss)
        
        print("Testing")
        test_loss = run_epoch(test_loader, model, loss, optimizer)
        print(test_loss)

        wandb.log({"Epoch": i})
    
if __name__ == "__main__":
    main()