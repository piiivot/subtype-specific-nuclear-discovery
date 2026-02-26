"""
Everything needed to run MIME training:

  Utilities     set_seed, save_checkpoint, load_checkpoint
  FileLogger    logs to stdout and a file
  MIMEEvaluator evaluates a model on a DataLoader (reports loss and MRR)
  MIMETrainer   full MIME training loop (warm-up + pseudo-label updates)
"""

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from transform import BasicImageTransform


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────

def set_seed(seed: int, deterministic: bool = True):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, base_seed: int = 0):
    """DataLoader worker seed initializer."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, path: Path, config: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    if config:
        ckpt["config"] = config
    torch.save(ckpt, path)


def load_checkpoint(model: nn.Module, path: Path, optimizer=None, device: str = "cuda") -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ─────────────────────────────────────────
# Logger
# ─────────────────────────────────────────

class FileLogger:
    """
    Writes timestamped log messages to stdout and to a log file.

    Output directory: ``output_dir`` if provided, otherwise
    ``output/mime/{timestamp}``.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.output_dir / "log.txt"
        else:
            self.output_dir = Path(f"output/mime/{ts}")
            log_dir = self.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "train.log"

        self._fh = open(log_path, "w", encoding="utf-8")
        self.info(f"Log file: {log_path}")

    def info(self, msg: str):
        self._write(msg)

    def warning(self, msg: str):
        self._write(f"WARNING: {msg}")

    def error(self, msg: str):
        self._write(f"ERROR: {msg}")

    def _write(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        sys.stdout.flush()
        self._fh.write(line + "\n")
        self._fh.flush()

    def get_output_dir(self) -> Path:
        return self.output_dir

    def close(self):
        if hasattr(self, "_fh") and not self._fh.closed:
            self._fh.close()

    def __del__(self):
        self.close()


# ─────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────

class MIMEEvaluator:
    """
    Evaluates a model on a DataLoader.

    Handles both single-label (int) and multi-label (one-hot float) targets:
    single-label targets are kept as-is for MRR computation; multi-label
    targets are used to compute the loss but argmax is used for MRR.
    """

    def __init__(self, model: nn.Module, loss_fn=None, device: str = "cuda"):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_labels, all_logits = [], []
        total_loss, n_batches = 0.0, 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                logits = outputs["logits"]

                if self.loss_fn is not None:
                    # Convert single-label to one-hot if needed
                    if labels.dim() == 1:
                        labels_onehot = torch.zeros(
                            labels.size(0), logits.size(1), device=self.device
                        )
                        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
                        loss = self.loss_fn(outputs, labels_onehot)
                    else:
                        loss = self.loss_fn(outputs, labels)
                    total_loss += loss.item()
                    n_batches += 1

                all_labels.append(
                    labels.cpu() if labels.dim() == 1 else labels.argmax(dim=1).cpu()
                )
                all_logits.append(logits.cpu())

        labels_cat = torch.cat(all_labels)
        logits_cat = torch.cat(all_logits)

        p = torch.sigmoid(logits_cat)
        sorted_idx = torch.argsort(-p, dim=1)
        ranks = (sorted_idx == labels_cat.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        results = {"MRR": (1.0 / ranks.float()).mean().item()}

        if self.loss_fn is not None and n_batches > 0:
            results["loss"] = total_loss / n_batches
        return results

    @staticmethod
    def get_plot_keys() -> List[str]:
        return ["loss", "MRR"]


# ─────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────

class MIMETrainer:
    """
    MIME training loop with pseudo-label refinement.

    Algorithm
    ---------
    1. **Warm-up** (``warm_up_epochs`` epochs):
       Train on the MIME dataset whose pseudo-labels are initialised as the
       observed single-class labels (one-hot).

    2. **Main phase** (remaining epochs):
       Each epoch:
         a. Train for ``steps_per_epoch`` random mini-batches.
         b. Evaluate on the val / train-eval sets.
         c. Sweep the full training set (without augmentation) and add class j
            to the pseudo-label of sample i whenever

                log p(yᵢⱼ=1|xᵢ) − log p(yᵢⱼ=0|xᵢ) > tau

            for any class j not already positive.

    The evaluation DataLoaders use the single-label version of the dataset
    (``EightClassDatasetKSplit``) so that standard accuracy / set-size metrics
    can be computed against the observed ground-truth labels.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer,
        optimizer_wrapper,
        logger: FileLogger,
        device: str = "cuda",
        save_dir: Optional[Path] = None,
        evaluator: Optional[MIMEEvaluator] = None,
        # MIME-specific
        warm_up_epochs: int = 5,
        tau: float = 0.3,
        num_classes: int = 8,
        steps_per_epoch: int = 1000,
        pseudo_label_update_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_wrapper = optimizer_wrapper
        self.logger = logger
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.evaluator = evaluator
        self.warm_up_epochs = warm_up_epochs
        self.tau = tau
        self.num_classes = num_classes
        self.steps_per_epoch = steps_per_epoch
        self.pseudo_label_update_batch_size = pseudo_label_update_batch_size
        self.eval_batch_size = eval_batch_size

        self.metric_history: List[Dict] = []
        self.output_dir: Path = logger.get_output_dir()
        self.total_updates_per_class = np.zeros(num_classes, dtype=np.int64)

        # Internal state for random-batch iterator
        self._train_iter: Optional[Iterator] = None
        self._iter_loader: Optional[DataLoader] = None

    # ─── Public entry point ──────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        train_eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 60,
        save_interval: int = 1,
    ):
        mime_dataset = train_loader.dataset
        if not hasattr(mime_dataset, "update_pseudo_label"):
            raise ValueError("train_loader.dataset must be an EightClassDatasetKSplitMIME instance.")

        # Replace MIME loaders with single-label loaders for evaluation
        self.logger.info("[MIME] Creating single-label evaluation DataLoaders...")
        if train_eval_loader is not None:
            train_eval_loader = self._make_eval_loader(train_eval_loader, "train_eval")
        if val_loader is not None:
            val_loader = self._make_eval_loader(val_loader, "val")

        # Log configuration
        self._log_config(num_epochs, train_loader.batch_size, save_interval)

        # ── Warm-up phase ────────────────────────────────────────────
        self.logger.info(f"\n=== Warm-Up Phase (epochs 1–{self.warm_up_epochs}) ===")
        for epoch in range(1, self.warm_up_epochs + 1):
            self._run_epoch(epoch, train_loader, val_loader, train_eval_loader, save_interval)

        self._log_pseudo_label_stats(mime_dataset, "After warm-up")

        # ── Main phase ───────────────────────────────────────────────
        self.logger.info(f"\n=== Main Phase (epochs {self.warm_up_epochs + 1}–{num_epochs}) ===")
        for epoch in range(self.warm_up_epochs + 1, num_epochs + 1):
            self._run_epoch(epoch, train_loader, val_loader, train_eval_loader, save_interval)

            # Update pseudo-labels after each main-phase epoch
            self.logger.info("\n--- Updating Pseudo-Labels ---")
            n_updated = self._update_pseudo_labels(mime_dataset, train_loader.batch_size)
            self.logger.info(f"  Updated {n_updated} labels this epoch")
            self.logger.info(f"  Cumulative updates per class: {self.total_updates_per_class.tolist()}")
            self._log_pseudo_label_stats(mime_dataset)

        self.logger.info("\n=== Training Completed ===")

    # ─── Epoch ───────────────────────────────────────────────────────

    def _run_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        train_eval_loader: Optional[DataLoader],
        save_interval: int,
    ):
        self.logger.info(f"\n=== Epoch {epoch} ===")
        train_loss = self._train_epoch(train_loader)
        epoch_metrics: Dict = {"epoch": epoch, "train_loss": train_loss}

        if self.evaluator is not None:
            eval_loader = train_eval_loader if train_eval_loader is not None else train_loader
            self.logger.info("\n--- Training Data Evaluation ---")
            train_results = self.evaluator.evaluate(eval_loader)
            self._log_and_store(train_results, "train", epoch_metrics)

            if val_loader is not None:
                self.logger.info("\n--- Validation ---")
                val_results = self.evaluator.evaluate(val_loader)
                self._log_and_store(val_results, "val", epoch_metrics)

            self.model.train()

        self.metric_history.append(epoch_metrics)
        self._save_metrics()
        self._save_plots(epoch)

        if epoch % save_interval == 0:
            self._save_checkpoint(epoch, train_loss)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Random-batch training: draw ``steps_per_epoch`` batches."""
        self.model.train()
        if self._iter_loader is not train_loader:
            self._iter_loader = train_loader
            self._train_iter = iter(train_loader)

        total_loss = 0.0
        for step in range(self.steps_per_epoch):
            try:
                images, labels = next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(train_loader)
                images, labels = next(self._train_iter)

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                self.logger.info(f"  Step [{step + 1}/{self.steps_per_epoch}]  loss={loss.item():.4f}")

        return total_loss / self.steps_per_epoch

    # ─── Pseudo-label update ─────────────────────────────────────────

    def _update_pseudo_labels(self, mime_dataset, batch_size: int) -> int:
        """
        Sweep the full training set and add class j to sample i's pseudo-label
        if  log p(yⱼ=1|xᵢ) − log p(yⱼ=0|xᵢ) > tau,  i.e.  logit_j > tau.

        The dataset transform is temporarily replaced with BasicImageTransform
        for faster inference (no stain augmentation needed here).
        """
        update_bs = self.pseudo_label_update_batch_size or batch_size

        orig_transform = mime_dataset.transform
        mime_dataset.transform = BasicImageTransform()

        loader = DataLoader(
            mime_dataset,
            batch_size=update_bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        self.model.eval()
        n_updated = 0
        self.logger.info(f"  Scanning {len(loader)} batches (batch_size={update_bs})...")

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                    self.logger.info(
                        f"  [{batch_idx + 1}/{len(loader)}] "
                        f"({100 * (batch_idx + 1) / len(loader):.1f}%)"
                    )

                images = images.to(self.device)
                logits = self.model(images)["logits"]  # [B, C]

                # log p(1) - log p(0) = log_sigmoid(logit) - log_sigmoid(-logit)
                log_diff = F.logsigmoid(logits) - F.logsigmoid(-logits)
                log_diff_np = log_diff.cpu().numpy()

                start = batch_idx * update_bs
                end   = min(start + images.size(0), len(mime_dataset))
                cur_labels = mime_dataset.pseudo_labels[start:end]

                # Add label j for sample i if currently 0 and log-odds > tau
                mask = (cur_labels == 0.0) & (log_diff_np > self.tau)
                mime_dataset.pseudo_labels[start:end][mask] = 1.0

                for c in range(self.num_classes):
                    self.total_updates_per_class[c] += int(mask[:, c].sum())

                n_updated += int(mask.sum())

        self.model.train()
        mime_dataset.transform = orig_transform
        return n_updated

    # ─── Helpers ─────────────────────────────────────────────────────

    def _make_eval_loader(self, loader: DataLoader, name: str) -> DataLoader:
        """Replace a MIME DataLoader with a single-label DataLoader."""
        from dataset import EightClassDatasetKSplit

        # Unwrap Subset if present
        if isinstance(loader.dataset, Subset):
            mime_ds = loader.dataset.dataset
            indices = loader.dataset.indices
        else:
            mime_ds = loader.dataset
            indices = None

        if not hasattr(mime_ds, "update_pseudo_label"):
            self.logger.info(f"[MIME] {name}_loader already uses single-label dataset")
            return loader

        normal_ds = EightClassDatasetKSplit(
            data_path=mime_ds.data_path,
            fold_csv_path=mime_ds.fold_csv_path,
            use_folds=mime_ds.use_folds,
            transform=mime_ds.transform,
            use_pickle=mime_ds.use_pickle,
            patch_size=mime_ds.patch_size,
            use_cache=True,
        )
        if indices is not None:
            normal_ds = Subset(normal_ds, indices)

        bs = self.eval_batch_size or loader.batch_size
        new_loader = DataLoader(normal_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        self.logger.info(f"[MIME] Created {name}_loader: {len(new_loader.dataset)} samples (bs={bs})")
        return new_loader

    def _log_pseudo_label_stats(self, mime_dataset, label: str = ""):
        stats = mime_dataset.get_pseudo_label_stats()
        prefix = f"[{label}] " if label else ""
        self.logger.info(
            f"  {prefix}avg labels/sample: {stats['avg_num_labels']:.4f} "
            f"± {stats['std_num_labels']:.4f}"
        )

    def _log_and_store(self, results: Dict, split: str, epoch_metrics: Dict):
        if "loss" in results:
            self.logger.info(f"  loss: {results['loss']:.4f}")
            epoch_metrics[f"{split}_loss"] = results["loss"]
        for k, v in results.items():
            if k == "loss":
                continue
            if isinstance(v, float):
                self.logger.info(f"  {k}: {v:.4f}")
                epoch_metrics[f"{split}_{k}"] = v

    def _log_config(self, num_epochs: int, batch_size: int, save_interval: int):
        self.logger.info("=== Training Configuration ===")
        cfg = {
            "trainer": "MIMETrainer",
            "warm_up_epochs": self.warm_up_epochs,
            "tau": self.tau,
            "steps_per_epoch": self.steps_per_epoch,
            "num_classes": self.num_classes,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "save_interval": save_interval,
            "device": self.device,
        }
        if self.optimizer_wrapper is not None and hasattr(self.optimizer_wrapper, "as_config"):
            cfg.update(self.optimizer_wrapper.as_config())
        if hasattr(self.loss_fn, "as_config"):
            cfg.update(self.loss_fn.as_config())
        for k, v in cfg.items():
            self.logger.info(f"  {k}: {v}")

    def _save_checkpoint(self, epoch: int, loss: float):
        path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint(self.model, self.optimizer, epoch, loss, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def _save_metrics(self):
        if not self.metric_history:
            return
        csv_path  = self.output_dir / "metrics.csv"
        json_path = self.output_dir / "metrics.json"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.metric_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metric_history)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metric_history, f, indent=2)

    def _save_plots(self, epoch: int):
        if not self.metric_history:
            return
        plot_keys = self.evaluator.get_plot_keys() if self.evaluator else []
        if not plot_keys:
            return

        epochs = [m["epoch"] for m in self.metric_history]
        n = len(plot_keys)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes_flat = np.array(axes).flatten()

        for i, key in enumerate(plot_keys):
            ax = axes_flat[i]
            for split, color in [("train", "blue"), ("val", "orange")]:
                vals = [m.get(f"{split}_{key}") for m in self.metric_history]
                vals = [v for v in vals if v is not None]
                if vals:
                    ax.plot(epochs[: len(vals)], vals, label=split, color=color, marker="o")
            ax.set_title(key)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        plot_dir = self.save_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"metrics_epoch_{epoch}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────
# Optimizer wrapper
# (thin wrapper that provides as_config() for logging)
# ─────────────────────────────────────────

class AdamWOptimizer:
    def __init__(self, lr: float, weight_decay: float, **kwargs):
        self.lr = lr
        self.weight_decay = weight_decay

    def build(self, model: nn.Module):
        return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def as_config(self) -> dict:
        return {"optimizer": "AdamW", "lr": self.lr, "weight_decay": self.weight_decay}


