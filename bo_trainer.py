from trainer import run_epoch, run_test
from DataLoader.DataLoader import DataLoad
from painn import painn
from datetime import datetime
import argparse
import wandb
import torch
from tqdm import tqdm

from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler

enable_wandb = False

parser = argparse.ArgumentParser(description="Example script with CLI arguments.")
parser.add_argument(
    "--target_index", type=int, default=0, help="Index for target label"
)

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

def run_training(config, Target_index):
    Target_label = target_dict[Target_index]
    train_loader, test_loader, val_loader = DataLoad(
        batch_size=config["batch_size"], target_index=Target_index
    )
    model = painn()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config['weight_decay'])
    EPOCHS = config["epochs"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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

def main():
    args = parser.parse_args()
    Target_index = args.target_index
    # Target_index = 0
    Target_label = target_dict[Target_index]
    print(f"Target label: {Target_label}")

    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "epochs": 1,
        "batch_size": tune.choice([2, 4, 8, 16]),
        "target_label": Target_label,
        "smoothing_factor": tune.uniform(0.0,1.0),
        "plateau_decay": tune.uniform(0.0,1.0),
        "patience": tune.uniform(0.0,10.0),
        "datetime": datetime.now(),
        'weight_decay': tune.loguniform(1e-4, 1e-1),
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
    BO_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=config["epochs"],
        grace_period=1,
        reduction_factor=2,
    )
    
    result = tune.run(
        partial(config, Target_index),
        scheduler=BO_scheduler,
        config=config,
        num_samples=10
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")