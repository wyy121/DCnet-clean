import os
from contextlib import nullcontext
from typing import Callable, Optional

import hydra
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from data2 import get_qclevr_dataloaders
#from data import get_qclevr_dataloaders
from model_extracted_fb_crosscov import Conv2dEIRNN
#from model_318 import Conv2dEIRNN
from utils import AttrDict, seed
import numpy as np
import matplotlib.pyplot as plt


def unpack_batch(batch):
    if len(batch) == 3:
        cue, mixture, labels = batch
        return cue, mixture, labels, None
    if len(batch) == 4:
        cue, mixture, labels, modes = batch
        return cue, mixture, labels, modes
    if len(batch) == 5:
        cue, mixture, labels, _, modes = batch
        return cue, mixture, labels, modes
    raise ValueError(f"Unexpected batch structure with {len(batch)} elements")


def compute_eval_outputs_and_losses(config, model, criterion, cue, mixture, labels):
    outputs = model(cue, mixture, all_timesteps=config.criterion.all_timesteps)
    if config.criterion.all_timesteps:
        per_timestep_losses = [
            F.cross_entropy(output, labels, reduction="none") for output in outputs
        ]
        per_sample_loss = torch.stack(per_timestep_losses, dim=0).mean(dim=0)
        loss = per_sample_loss.mean()
        outputs = outputs[-1]
    else:
        loss = criterion(outputs, labels)
        per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")
    return outputs, loss, per_sample_loss

def train_iter(
    config: AttrDict,
    model: Conv2dEIRNN,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, float]:
    """
    Perform a single training iteration.

    Args:
        config (AttrDict): Configuration parameters.
        model (Conv2dEIRNN): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        wandb_log (Callable[[dict[str, float, int]], None]): Function to log training statistics to Weights & Biases.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple[float, float]: A tuple containing the training loss and accuracy.
    """
    if config.train.grad_clip.disable:
        clip_grad_ = lambda x, y: None
    elif config.train.grad_clip.type == "norm":
        clip_grad_ = lambda x, y: clip_grad_norm_(x, y, foreach=False)
    elif config.train.grad_clip.type == "value":
        clip_grad_ = lambda x, y: clip_grad_value_(x, y, foreach=False)
    else:
        raise NotImplementedError(
            f"Gradient clipping type {config.train.grad_clip_type} not implemented"
        )
    model.train()
    accumulation_steps = max(1, config.train.accumulation_steps)
    use_amp = bool(config.train.amp and device.type == "cuda")
    use_non_blocking = device.type == "cuda"
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    bar = tqdm(
        train_loader,
        desc=(f"Training | Epoch: {epoch} | " f"Loss: {0:.4f} | " f"Acc: {0:.2%}"),
        disable=not config.tqdm,
    )
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(bar):
        cue, mixture, labels, _ = unpack_batch(batch)
        cue = cue.to(device, non_blocking=use_non_blocking)
        mixture = mixture.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        # Forward pass
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast_context:
            outputs = model(
                cue, mixture, all_timesteps=config.criterion.all_timesteps
            )
            if config.criterion.all_timesteps:
                losses = []
                for output in outputs:
                    losses.append(criterion(output, labels))
                loss = sum(losses) / len(losses)
                outputs = outputs[-1]
            else:
                loss = criterion(outputs, labels)

        # Backward and optimize
        train_loss += loss.item()
        scaled_loss = loss / accumulation_steps
        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (
            (i + 1) % accumulation_steps == 0
            or i + 1 == len(train_loader)
        )
        if should_step:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            clip_grad_(model.parameters(), config.train.grad_clip.value)
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        # Update statistics
        running_loss += loss.item()

        predicted = outputs.argmax(-1)
        correct = (predicted == labels).sum().item()
        train_correct += correct
        running_correct += correct
        train_total += len(labels)
        running_total += len(labels)

        # Log statistics
        if (i + 1) % config.train.log_freq == 0:
            running_loss /= config.train.log_freq
            running_acc = running_correct / running_total
            wandb_log(dict(running_loss=running_loss, running_acc=running_acc))
            bar.set_description(
                f"Training | Epoch: {epoch} | "
                f"Loss: {running_loss:.4f} | "
                f"Acc: {running_acc:.2%}"
            )
            running_loss = 0
            running_correct = 0
            running_total = 0

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    wandb_log(dict(train_loss=train_loss, train_acc=train_acc))

    return train_loss, train_acc


def eval_iter(
    config: AttrDict,
    model: Conv2dEIRNN,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Perform a single evaluation iteration and report metrics by mode."""
    model.eval()
    use_non_blocking = device.type == "cuda"
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    mode_stats = {
        mode: {"correct": 0, "total": 0, "loss": 0.0}
        for mode in ["color", "shape", "conjunction"]
    }

    with torch.no_grad():
        for batch in val_loader:
            cue, mixture, labels, modes = unpack_batch(batch)
            cue = cue.to(device, non_blocking=use_non_blocking)
            mixture = mixture.to(device, non_blocking=use_non_blocking)
            labels = labels.to(device, non_blocking=use_non_blocking)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if config.train.amp and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                outputs, loss, per_sample_loss = compute_eval_outputs_and_losses(
                    config, model, criterion, cue, mixture, labels
                )

            test_loss += loss.item()
            predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            test_correct += correct
            test_total += len(labels)

            if modes is not None:
                for idx, mode in enumerate(modes):
                    if mode not in mode_stats:
                        continue
                    mode_stats[mode]["correct"] += int(predicted[idx] == labels[idx])
                    mode_stats[mode]["total"] += 1
                    mode_stats[mode]["loss"] += float(per_sample_loss[idx].item())

    test_loss /= len(val_loader)
    test_acc = test_correct / test_total

    metrics = dict(test_loss=test_loss, test_acc=test_acc, epoch=epoch)
    summarized_mode_stats = {}
    for mode, stats in mode_stats.items():
        if stats["total"] == 0:
            continue
        mode_acc = stats["correct"] / stats["total"]
        mode_loss = stats["loss"] / stats["total"]
        metrics[f"test_acc_{mode}"] = mode_acc
        metrics[f"test_loss_{mode}"] = mode_loss
        summarized_mode_stats[mode] = {
            "acc": mode_acc,
            "loss": mode_loss,
            "total": stats["total"],
        }

    wandb_log(metrics)

    return test_loss, test_acc, summarized_mode_stats


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str): Path to save the plot image
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config1.yaml",
)
def train(config: DictConfig) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (dict): Configuration parameters.
    """
    config = OmegaConf.to_container(config, resolve=True)
    config = AttrDict(config)
    # Set the random seed
    if config.seed is not None:
        seed(
            config.seed,
            deterministic=config.train.deterministic,
            cudnn_benchmark=config.train.cudnn_benchmark,
            allow_tf32=config.train.allow_tf32,
        )
    # Set the matmul precision
    torch.set_float32_matmul_precision(config.train.matmul_precision)
    # Get device and initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv2dEIRNN(**config.model).to(device)
    if getattr(getattr(model, "_orig_mod", model), "extracted_fb_enabled", False):
        print("Using fixed extracted rank-one feedback vectors.")

    # Compile the model if requested
    model = torch.compile(
        model,
        fullgraph=config.compile.fullgraph,
        dynamic=config.compile.dynamic,
        backend=config.compile.backend,
        mode=config.compile.mode,
        disable=config.compile.disable,
    )

    scaler_enabled = config.train.amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    # Initialize the optimizer
    if config.optimizer.fn == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    elif config.optimizer.fn == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    elif config.optimizer.fn == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer.fn} not implemented")

    # Initialize the loss function
    if config.criterion.fn == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {config.criterion.fn} not implemented")

    # Get the data loaders
    train_loader, val_loader = get_qclevr_dataloaders(
        data_root=config.data.root,
        assets_path=config.data.assets_path,
        train_batch_size=config.data.batch_size,
        val_batch_size=config.data.val_batch_size,
        resolution=config.model.input_size,
        holdout=config.data.holdout,
        mode=config.data.mode,
        primitive=config.data.primitive,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )

    if config.verbose:
        train_labels = torch.tensor(train_loader.dataset.counts)
        print("原始训练集类别分布:", torch.bincount(train_labels))
        val_labels = torch.tensor(val_loader.dataset.counts)
        print("原始验证集类别分布:", torch.bincount(val_labels))
        print(f"训练集总样本数: {len(train_loader.dataset)}")
        print(f"验证集总样本数: {len(val_loader.dataset)}")

    # Initialize the learning rate scheduler
    if config.scheduler.fn is None:
        scheduler = None
    if config.scheduler.fn == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.optimizer.lr,
            total_steps=config.train.epochs * len(train_loader),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler.fn} not implemented")

    # Initialize Weights & Biases
    if config.wandb:
        wandb.init(project="EI RNN", config=config)
        wandb_log = lambda x: wandb.log(x)
    else:
        wandb_log = lambda x: None

    # Create the checkpoint directory
    if config.wandb:
        checkpoint_dir = os.path.join(config.checkpoint.root, wandb.run.name)
    else:
        checkpoint_dir = config.checkpoint.root
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': []
    }

    for epoch in range(config.train.epochs):
        # Train the model
        train_loss, train_acc = train_iter(
            config,
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            wandb_log,
            epoch,
            device,
            scaler,
        )

        # Evaluate the model on the validation set
        test_loss, test_acc, mode_metrics = eval_iter(
            config, model, criterion, val_loader, wandb_log, epoch, device
        )

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch)

        # Print the epoch statistics
        mode_metrics_str = " | ".join(
            [
                f"{mode}: {stats['acc']:.2%}"
                for mode, stats in mode_metrics.items()
            ]
        )
        print(
            f"Epoch [{epoch}/{config.train.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc:.2%}"
            + (f" | {mode_metrics_str}" if mode_metrics_str else "")
        )

        # Save the model
        file_path = os.path.abspath(
            os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        )
        link_path = os.path.abspath(os.path.join(checkpoint_dir, "checkpoint.pt"))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history  # 保存训练历史
        }
        torch.save(checkpoint, file_path)
        import shutil
        try:
            os.remove(link_path)
        except FileNotFoundError:
            pass
        shutil.copy2(file_path, link_path)

        # Plot training curves every few epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == config.train.epochs - 1:
            plot_save_path = os.path.join(checkpoint_dir, f"training_curves_epoch_{epoch}.png")
            plot_training_curves(history, plot_save_path)

    # Final plot
    final_plot_path = os.path.join(checkpoint_dir, "final_training_curves.png")
    plot_training_curves(history, final_plot_path)
    
    # Save history to file
    history_path = os.path.join(checkpoint_dir, "training_history.npy")
    np.save(history_path, history)
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    train()
    
