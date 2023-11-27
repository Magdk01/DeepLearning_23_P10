from painn import painn
from DataLoader.DataLoader import DataLoad
import torch
from tqdm import tqdm
import numpy as np

import wandb
from datetime import datetime
from collections.abc import Iterable

import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau


parser = argparse.ArgumentParser(description="Example script with CLI arguments.")
parser.add_argument(
    "--target_index", type=int, default=0, help="Index for target label"
)


enable_wandb = False


def run_epoch(loader, model, loss_fn, optimizer, scheduler, config, val_loader=None):
    def extract_and_calc_loss(x, inner_batch_indexies):
        atomic_numbers, coords, y = x
        atomic_numbers = np.array(atomic_numbers)
        coords = np.array(coords)
        # y = y[0][0]

        # Make predictions for this batch
        outputs = model(atomic_numbers, coords, inner_batch_indexies)
        # assert not np.isnan(outputs.item())

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y)
        # Squared error as loss function
        # loss = (outputs - y) ** 2
        return loss

    smoothed_loss = 0.0
    alpha = config["smoothing_factor"]  # Smoothing factor

    total_iterations = len(loader)
    for i, (batch, batch_indexies) in tqdm(enumerate(loader), total=total_iterations):
        batch_save = batch
        concatenated_list = [
            torch.cat(elements) if not idx == 2 else torch.tensor([elements])
            for idx, elements in enumerate(zip(*batch))
        ]

        optimizer.zero_grad()
        loss = extract_and_calc_loss(concatenated_list, batch_indexies)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        if enable_wandb:
            wandb.log({"Train loss": loss})

        if (i + 1) % 10 == 0:
            avg_loss_val_list = []
            for val_i, (val_batch, val_batch_indexies) in enumerate(val_loader):
                val_concatenated_list = [
                    torch.cat(elements) if not idx == 2 else torch.tensor([elements])
                    for idx, elements in enumerate(zip(*val_batch))
                ]
                val_loss = extract_and_calc_loss(
                    val_concatenated_list, val_batch_indexies
                )
                avg_loss_val_list.append(float(val_loss))

            alvl = np.mean(avg_loss_val_list)
            smoothed_loss = (alpha * smoothed_loss) + ((1 - alpha) * alvl)
            scheduler.step(smoothed_loss)

            if enable_wandb:
                wandb.log({"Mean Validation loss": alvl, "i-th_timestep": i})
        break
    return model


target_dict = {
    0: "Dipole moment",
    1: "Isotropic polarizability",
    2: "Highest occupied molecular orbital energy",
    3: "Lowest unoccupied molecular orbital energy",
    4: "Gap between HOMO and LUMO",
    5: "Electronic Spatial extent",
    6: "Zero point vibrational energy",
    7: "Internal energy at 0K",
    8: "Internal energy at 298.15K",
    9: "Enthalpy at 298.15K",
    10: "Free energy at 298.15K",
    11: "Heat capacity at 298.15K",
}


def main():
    args = parser.parse_args()
    Target_index = args.target_index
    # Target_index = 0
    Target_label = target_dict[Target_index]
    print(f"Target label: {Target_label}")
    global enable_wandb
    enable_wandb = True

    config = {
        "learning_rate": 0.01,
        "epochs": 1,
        "batch_size": 8,
        "target_label": Target_label,
        "smoothing_factor": 0.9,
        "plateau_decay": 0.5,
        "patience": 5,
        "datetime": datetime.now(),
    }

    if enable_wandb:
        wandb.login()
        wandb.init(
            project="painn",
            entity="deep_learing_p10",
            name=f"Train run for {Target_label}. Datetime :{config['datetime']}",
            config=config,
        )

    train_loader, test_loader, val_loader = DataLoad(
        batch_size=config["batch_size"], target_index=Target_index
    )
    model = painn()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    EPOCHS = config["epochs"]
    scheduler = ReduceLROnPlateau(
        optimizer, factor=config["plateau_decay"], patience=config["patience"]
    )

    for i in tqdm(range(EPOCHS)):
        trained_model = run_epoch(
            train_loader,
            model,
            loss,
            optimizer,
            scheduler,
            config,
            val_loader=val_loader,
        )
        torch.save(trained_model, f"{Target_label}_model_{config['datetime']}.pth")
        if enable_wandb:
            wandb.log({"Epoch": i})


if __name__ == "__main__":
    main()
