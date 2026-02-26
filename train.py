"""
Training entry point for MIME
(Multi-label Inference with Model-based Estimation).

Usage:
    python train.py [options]   # see --help for full list
    bash train.sh               # convenience wrapper with default settings
"""

import argparse
import json
import os
import sys

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from dataset import EightClassDatasetKSplit, EightClassDatasetKSplitMIME
from model import ResNet18Backbone, MIMEHead, BaseNet
from loss import MIMELoss
from transform import AugmentedRandStainNAWithNoiseTransform, BasicImageTransform
from trainer import (
    set_seed, worker_init_fn, save_checkpoint, load_checkpoint,
    FileLogger, MIMEEvaluator, MIMETrainer,
    AdamWOptimizer,
)


# ─────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MIME")

    # Data
    p.add_argument("--data_path",   required=True,
                   help="Path to the LMDB directory (train / val share the same LMDB)")
    p.add_argument("--fold_csv_path", required=True,
                   help="Path to the k-fold split CSV "
                        "(columns: fold, case_id, class_name)")
    p.add_argument("--except_fold_idx", type=int, required=True,
                   help="Fold index used as the validation set")
    p.add_argument("--cal_fold_idx",    type=int, required=True,
                   help="Fold index used as the calibration set")
    p.add_argument("--num_folds",   type=int, default=10)
    p.add_argument("--use_pickle",  type=lambda x: x.lower() == "true", default=True,
                   help="True if LMDB values are pickle-serialised (default: True)")
    p.add_argument("--patch_size",  type=int, default=64)

    # Training
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_epochs",   type=int, default=60)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       type=str, default="cuda:0")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--save_interval",type=int, default=1,
                   help="Save a checkpoint every N epochs")

    # Optimizer
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay",  type=float, default=1e-5)

    # Model
    p.add_argument("--pretrained",   type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--num_classes",  type=int,   default=8)
    p.add_argument("--in_features",  type=int,   default=512,
                   help="Backbone output dimension (512 for ResNet-18)")
    p.add_argument("--latent_dim",   type=int,   default=512)

    # Loss
    p.add_argument("--kl_weight",    type=float, default=1e-3,
                   help="Weight β on the KL regularization term")

    # MIME
    p.add_argument("--warm_up_epochs",  type=int,   default=5)
    p.add_argument("--tau",             type=float, default=0.3,
                   help="Log-odds threshold for pseudo-label updates")
    p.add_argument("--steps_per_epoch", type=int,   default=1000,
                   help="Number of mini-batch steps per epoch")
    p.add_argument("--pseudo_label_update_batch_size", type=int, default=4096,
                   help="Batch size used when sweeping data for pseudo-label update")
    p.add_argument("--eval_batch_size", type=int, default=4096,
                   help="Batch size used during evaluation")

    # Output
    p.add_argument("--output_dir",   type=str, default=None,
                   help="Directory for checkpoints and logs "
                        "(auto-generated if not specified)")

    # Optional checkpoint initialisation
    p.add_argument("--init_checkpoint",         type=str, default=None,
                   help="Load full model weights from this checkpoint")
    p.add_argument("--backbone_init_checkpoint",type=str, default=None,
                   help="Load backbone weights only from this checkpoint")

    return p.parse_args()


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    args = parse_args()

    set_seed(args.seed, deterministic=True)
    print(f"Random seed: {args.seed}")

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
        print("Warning: CUDA not available, using CPU.")

    # ── Logger ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else None
    logger = FileLogger(output_dir=output_dir)
    output_dir = logger.get_output_dir()

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # ── Transforms ────────────────────────────────────────────────────
    train_transform = AugmentedRandStainNAWithNoiseTransform()
    val_transform = BasicImageTransform()

    # ── Datasets ──────────────────────────────────────────────────────
    dataset_kwargs = dict(
        data_path=args.data_path,
        fold_csv_path=args.fold_csv_path,
        except_fold_idx=args.except_fold_idx,
        cal_fold_idx=args.cal_fold_idx,
        num_folds=args.num_folds,
        use_pickle=args.use_pickle,
        patch_size=args.patch_size,
    )

    # Training dataset (MIME, with pseudo-labels and augmentation)
    train_dataset = EightClassDatasetKSplitMIME(
        transform=train_transform, mode="train", **dataset_kwargs
    )

    # Validation dataset (single-label; will be replaced inside the trainer)
    val_dataset = EightClassDatasetKSplitMIME(
        transform=val_transform, mode="val", **dataset_kwargs
    )

    # Train-eval dataset: 50 K random samples without augmentation
    train_dataset_for_eval = EightClassDatasetKSplitMIME(
        transform=val_transform, mode="train", **dataset_kwargs
    )
    rng = np.random.RandomState(args.seed)
    n_eval = min(50_000, len(train_dataset_for_eval))
    eval_indices = rng.choice(len(train_dataset_for_eval), size=n_eval, replace=False)
    train_eval_dataset = Subset(train_dataset_for_eval, eval_indices)

    logger.info(f"Train size:      {len(train_dataset)}")
    logger.info(f"Val size:        {len(val_dataset)}")
    logger.info(f"Train-eval size: {len(train_eval_dataset)}")

    # ── DataLoaders ───────────────────────────────────────────────────
    init_fn = lambda wid: worker_init_fn(wid, args.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=init_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────
    backbone = ResNet18Backbone(pretrained=args.pretrained)
    head = MIMEHead(
        in_features=args.in_features,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim,
    )
    model = BaseNet(backbone=backbone, heads=head).to(device)

    if args.init_checkpoint:
        logger.info(f"Loading model weights from {args.init_checkpoint}")
        load_checkpoint(model, Path(args.init_checkpoint), device=device)

    elif args.backbone_init_checkpoint:
        logger.info(f"Loading backbone weights from {args.backbone_init_checkpoint}")
        ckpt = torch.load(args.backbone_init_checkpoint, map_location=device)
        model.backbone.load_state_dict(ckpt["model_state_dict"], strict=False)

    # ── Optimizer ─────────────────────────────────────────────────────
    opt_wrapper = AdamWOptimizer(lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = opt_wrapper.build(model)

    # ── Loss & Evaluator ──────────────────────────────────────────────
    loss_fn = MIMELoss(kl_weight=args.kl_weight)
    evaluator = MIMEEvaluator(model=model, loss_fn=loss_fn, device=device)

    # ── Trainer ───────────────────────────────────────────────────────
    save_dir = output_dir / "checkpoints"
    trainer = MIMETrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        optimizer_wrapper=opt_wrapper,
        logger=logger,
        device=device,
        save_dir=save_dir,
        evaluator=evaluator,
        warm_up_epochs=args.warm_up_epochs,
        tau=args.tau,
        num_classes=args.num_classes,
        steps_per_epoch=args.steps_per_epoch,
        pseudo_label_update_batch_size=args.pseudo_label_update_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        train_eval_loader=train_eval_loader,
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
    )

    logger.info("Done.")
    logger.close()


if __name__ == "__main__":
    main()
