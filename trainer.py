from painn import painn
from DataLoader.DataLoader import DataLoad
import torch
from tqdm import tqdm
import numpy as np

import wandb
from datetime import datetime

# WandB setup
config = {
    "learning_rate": 0.001,
    "epochs": 1,
    "batch_size": 1,
}

enable_wandb = True
if enable_wandb:
    wandb.login()
    wandb.init(
        project="painn",
        entity="deep_learing_p10",
        name=f"Run_at:{datetime.now()}",
        config=config,
    )


def run_epoch(loader, model, loss_fn, optimizer, batch_indexies, val_loader=None):
    def extract_and_calc_loss(x):
        atomic_numbers, coords, y = x
        atomic_numbers = np.array(atomic_numbers)
        coords = np.array(coords)
        y = y[0][0]

        # Make predictions for this batch
        outputs = model(atomic_numbers, coords, batch_indexies)
        # assert not np.isnan(outputs.item())

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y)
        # Squared error as loss function
        # loss = (outputs - y) ** 2
        return loss

    for i, (batch, batch_indexies) in enumerate(loader):
        concatenated_list = [torch.cat(elements) for elements in zip(*batch)]
        # for j, x in enumerate(batch):
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss = extract_and_calc_loss(concatenated_list)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        if enable_wandb:
            wandb.log({"Train loss": loss})

        if i + 1 % 1000 == 0 and val_loader:
            # print(i)
            avg_loss_val_list = []
            val_load = val_loader.__iter__()
            for _ in range(100):
                avg_loss_val_list.append(extract_and_calc_loss(next(val_load)[0]))

            avg_loss_val_list = torch.tensor(avg_loss_val_list)
            print(f"Mean Validation loss: {(alvl:=torch.mean(avg_loss_val_list))}")

            if enable_wandb:
                wandb.log({"Mean Validation loss": alvl, "i-th_timestep": i})

    return True


def main():
    train_loader, test_loader, val_loader = DataLoad(batch_size=4)
    model = painn()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    EPOCHS = config["epochs"]

    for i in tqdm(range(EPOCHS)):
        # print("Training")
        train_loss = run_epoch(train_loader, model, loss, optimizer, val_loader)
        # print(train_loss)

        # print("Testing")
        test_loss = run_epoch(test_loader, model, loss, optimizer)
        # print(test_loss)

        if enable_wandb:
            wandb.log({"Epoch": i})


if __name__ == "__main__":
    main()
