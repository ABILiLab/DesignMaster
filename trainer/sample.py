#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from model_base.edm import EDM as EDM_backbone
from model_mc_v3.edm_mc_v3 import EDM_CFG
from trainer.trainer_mc_v3 import CFGTrainer
from utils.dataset import PROTACDataset, collate
from utils.utils import disable_rdkit_logging


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--geom_ckpt", type=str, default=str(ROOT / "checkpoints/geom_best.ckpt"))
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--context_nf", type=int, default=128)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--disable_cfg", action="store_true")
    p.add_argument("--test_prefix", type=str, default="protacs_test_with_props")
    p.add_argument("--save_prefix", type=str, default="designmaster")
    return p.parse_args()


def main():
    disable_rdkit_logging()
    args = parse_args()
    seed_everything(args.seed)
    guidance_scale = None if args.disable_cfg else args.guidance_scale
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backbone_model = EDM_backbone(
        device=device,
        in_node_nf=9,
        hidden_nf=128,
        ffn_embedding_dim=1024,
        attention_heads=32,
        n_layers=6,
        tanh=False,
        coords_range=10.0,
        dropout=0.0,
        activation_dropout=0.0,
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
        dropout=0.0,
        activation_dropout=0.0,
        backbone=backbone,
        guidance_scale=guidance_scale,
        condition_drop_prob=0.0 if args.disable_cfg else 0.1,
        context_nf=args.context_nf,
    )
    ckpt = torch.load(args.model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"load {args.model_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()

    test_ds = PROTACDataset(data_path=str(args.data_path), prefix=args.test_prefix)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    print(
        f"sample n_mol={len(test_ds)} n_samples={args.n_samples} "
        f"batch={args.batch_size} disable_cfg={args.disable_cfg}"
    )

    trainer = CFGTrainer(
        model=model,
        device=device,
        epochs=None,
        analyze_epochs=None,
        optimizer=None,
        run=None,
        loss_type="l2",
        save_path=str(args.output_dir),
        save_prefix=args.save_prefix,
        n_stability_samples=args.n_samples,
        guidance_scale=guidance_scale,
        property_adjustment=False,
    )
    trainer.pred(loader, str(args.output_dir), sample_fn=None, save_chains_for_viz=False)
    print("Sampling done.")


if __name__ == "__main__":
    main()
