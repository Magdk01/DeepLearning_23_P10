from painn import painn
from DataLoader.DataLoader import DataLoad
import torch
from tqdm import tqdm
import numpy as np

def run_epoch(loader, model, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.
    
    for i, batch in enumerate(loader):
        for j, x in enumerate(batch):
            atomic_numbers, coords, y = x
            
            y = y[0][0]

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(atomic_numbers.unsqueeze(1), coords)
            assert not np.isnan(outputs.item()) 

            # Compute the loss and its gradients
            loss = loss_fn(outputs, y)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            running_loss += loss.item()
            
            if j == len(batch) - 1:
                last_loss = running_loss / len(batch) # loss per batch
                print('batch {} loss: {} n atoms {}'.format(i + 1, last_loss, len(atomic_numbers)))
                running_loss = 0.

    return last_loss

def main():
    train_loader, test_loader, val_loader = DataLoad()
    model = painn()
    loss = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    EPOCHS = 1
    
    for _ in tqdm(range(EPOCHS)):
        print("Training")
        train_loss = run_epoch(train_loader, model, loss, optimizer)
        print(train_loss)
        
        print("Testing")
        test_loss = run_epoch(test_loader, model, loss, optimizer)
        print(test_loss)
    
if __name__ == "__main__":
    main()