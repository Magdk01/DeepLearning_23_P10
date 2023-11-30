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


global enable_wandb
enable_wandb = True

def extract_and_calc_loss(x, model, loss_fn, inner_batch_indexies):
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


def run_epoch(loader, model, loss_fn, optimizer, scheduler, config, val_loader=None):
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
        loss = extract_and_calc_loss(concatenated_list, model, loss_fn, batch_indexies)

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
                    val_concatenated_list, model, loss_fn, val_batch_indexies
                )
                avg_loss_val_list.append(float(val_loss))

            alvl = np.mean(avg_loss_val_list)
            smoothed_loss = (alpha * smoothed_loss) + ((1 - alpha) * alvl)
            scheduler.step(smoothed_loss)

            if enable_wandb:
                wandb.log({"Mean Validation loss": alvl, "i-th_timestep": i})
    return model

def run_test(test_loader, test_model, test_loss_fn):
    avg_loss_test_list = []
    for test_i, (test_batch, test_batch_indexies) in enumerate(test_loader):
        test_concatenated_list = [
            torch.cat(elements) if not idx == 2 else torch.tensor([elements])
            for idx, elements in enumerate(zip(*test_batch))
        ]
        test_loss = extract_and_calc_loss(
            test_concatenated_list, test_model, test_loss_fn, test_batch_indexies
        )
        avg_loss_test_list.append(float(test_loss))

    return np.mean(avg_loss_test_list)


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

    config = {
        "learning_rate": 0.01,
        "epochs": 1,
        "batch_size": 8,
        "target_label": Target_label,
        "smoothing_factor": 0.9,
        "plateau_decay": 0.5,
        "patience": 5,
        "datetime": datetime.now(),
        'weight_decay': 0.01,
        'swa_start': 2,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config['weight_decay'])
    EPOCHS = config["epochs"]
    scheduler = ReduceLROnPlateau(
        optimizer, factor=config["plateau_decay"], patience=config["patience"]
    )
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=config["learning_rate"])
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    for i in tqdm(range(EPOCHS)):
        current_scheduler = scheduler if i < config['swa_start'] else swa_scheduler
        trained_model = run_epoch(
            train_loader,
            model,
            loss,
            optimizer,
            current_scheduler,
            config,
            val_loader=val_loader,
        )
        if i > config['swa_start']:
            swa_model.update_parameters(model)
        
        torch.save(
            trained_model,
            f"{Target_label.replace(' ', '_').lower()}_model_{config['datetime']}.pth",
        )
        if enable_wandb:
            wandb.log({"Epoch": i})
    
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    test_loss = run_test(test_loader, swa_model, torch.nn.L1Loss())
    if enable_wandb:
        wandb.log({"Test Loss": test_loss})

if __name__ == "__main__":
    main()
