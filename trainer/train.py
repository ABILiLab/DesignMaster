#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT))

import wandb

from model_base.edm import EDM as EDM_backbone
from model_mc_v3.edm_mc_v3 import EDM_CFG
from trainer.trainer_mc_v3 import CFGTrainer
from utils.dataset import PROTACDataset, collate
from utils.utils import disable_rdkit_logging


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("2.0", "3.0"), required=True)
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--geom_ckpt", type=str, default=str(ROOT / "checkpoints/geom_best.ckpt"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--context_nf", type=int, default=128)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--condition_drop_prob", type=float, default=0.1)
    p.add_argument("--analyze_epochs", type=int, default=0)
    p.add_argument("--train_prefix", type=str, default="protacs_train_with_props")
    p.add_argument("--val_prefix", type=str, default="protacs_val_with_props")
    p.add_argument("--init_ckpt", type=Path, default=None)
    return p.parse_args()


def main():
    disable_rdkit_logging()
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoints.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"train dataset={args.dataset} "
        f"device={device} batch={args.batch_size} shuffle=False seed={args.seed}"
    )

    backbone_model = EDM_backbone(
        device=device,
        in_node_nf=9,
        hidden_nf=128,
        ffn_embedding_dim=1024,
        attention_heads=32,
        n_layers=6,
        tanh=False,
        coords_range=10.0,
        dropout=0.05,
        activation_dropout=0.05,
    )
    backbone_model.load_state_dict(torch.load(args.geom_ckpt, map_location="cpu"))
    backbone = backbone_model.dynamics.dynamics

    model = EDM_CFG(
        device=device,
        in_node_nf=9,
        hidden_nf=128,
        ffn_embedding_dim=1024,
        attention_heads=32,
        n_layers=6,
        tanh=False,
        coords_range=10.0,
        dropout=0.05,
        activation_dropout=0.05,
        backbone=backbone,
        guidance_scale=args.guidance_scale,
        condition_drop_prob=args.condition_drop_prob,
        context_nf=args.context_nf,
    )
    if args.init_ckpt is not None:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(
            f"init from {args.init_ckpt}: missing={len(missing)} unexpected={len(unexpected)}"
        )
    model = model.to(device)

    train_ds = PROTACDataset(data_path=str(args.data_path), prefix=args.train_prefix)
    val_ds = PROTACDataset(data_path=str(args.data_path), prefix=args.val_prefix)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )

    run = wandb.init(project="designmaster", config=vars(args), mode="disabled")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-12, amsgrad=True
    )
    trainer = CFGTrainer(
        model=model,
        device=device,
        epochs=args.epochs,
        analyze_epochs=args.analyze_epochs,
        optimizer=optimizer,
        run=run,
        loss_type="l2",
        save_path=str(args.checkpoints),
        save_prefix=args.exp_name,
        guidance_scale=args.guidance_scale,
        property_adjustment=False,
    )

    print("Start training ...")
    t0 = datetime.datetime.now()
    trainer.train(train_loader, val_loader)
    dt = (datetime.datetime.now() - t0).total_seconds() / 60.0
    best = args.checkpoints / f"{args.exp_name}_best.ckpt"
    print(f"Training done in {dt:.1f} min. best_loss={trainer.best_loss} best_epoch={trainer.best_epoch}")
    print(f"Best checkpoint: {best} exists={best.is_file()}")
    meta = args.checkpoints / f"{args.exp_name}_train_meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"dataset={args.dataset}",
                f"geom_ckpt={args.geom_ckpt}",
                f"shuffle=False",
                f"batch_size={args.batch_size}",
                f"epochs={args.epochs}",
                f"seed={args.seed}",
                f"context_nf={args.context_nf}",
                f"condition_drop_prob={args.condition_drop_prob}",
                f"train_prefix={args.train_prefix}",
                f"val_prefix={args.val_prefix}",
                f"data_path={args.data_path}",
                f"init_ckpt={args.init_ckpt}",
                f"best_loss={trainer.best_loss}",
                f"best_epoch={trainer.best_epoch}",
                f"elapsed_min={dt:.2f}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
