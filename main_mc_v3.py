import torch
import datetime
from torch.utils.data import DataLoader
import wandb
import argparse
from pathlib import Path

torch.set_printoptions(profile='full', linewidth=200)

from model_base.edm import EDM as EDM_backbone
from model_mc_v3.edm_mc_v3 import EDM_CFG
from utils.dataset import PROTACDataset, collate
from trainer.trainer_mc_v3 import CFGTrainer
from utils.utils import add_dict_to_argparser, disable_rdkit_logging


def create_argparser():
    defaults = dict(
        project='ours_base',
        exp_name='protacs_mc_la_20',
        log_dir='logs',
        data_path='datasets-2.0',
        train_data_prefix='protacs_train_self',
        val_data_prefix='protacs_val_self',
        test_data_prefix='protacs_test_self',
        checkpoints='checkpoints',
        epochs=200,
        batch_size=8,
        lr=5e-5,
        analyze_epochs=20,
        num_workers=16,
        seed=42,
        in_node_nf=9,
        enable_progress_bar=True,
        data_augmentation=False,
        resume=None,
    )
    egnn_defaults = dict(
        diffusion_steps=500,
        diffusion_noise_schedule='polynomial_2',
        diffusion_noise_precision=1e-5,
        diffusion_loss_type='l2',
        n_layers=6,
        hidden_nf=128,
        ffn_embedding_dim=1024,
        attention_heads=32,
        tanh=False,
        coords_range=10.,
        dropout=0.05,
        activation_dropout=0.05,
        # Classifier-free guidance specific parameters
        guidance_scale=3.0,  # Default guidance scale for sampling
        condition_drop_prob=0.1,  # Probability of dropping conditions during training
    )
    defaults.update(egnn_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main(args):
    run = wandb.init(
        project=args.project,
        config=args,
        mode='disabled'
    )
    rank = 'cuda'
    print("Preparing model and data ...")

    # Load the pre-trained backbone model
    model_2 = EDM_backbone(
        device=rank,
        in_node_nf=args.in_node_nf,
        hidden_nf=wandb.config.hidden_nf,
        ffn_embedding_dim=wandb.config.ffn_embedding_dim,
        attention_heads=wandb.config.attention_heads,
        n_layers=wandb.config.n_layers,
        tanh=wandb.config.tanh,
        coords_range=wandb.config.coords_range,
        dropout=wandb.config.dropout,
        activation_dropout=wandb.config.activation_dropout
    )
    model_2.load_state_dict(torch.load('checkpoints/geom_best.ckpt'))
    backbone = model_2.dynamics.dynamics
    # backbone = None
    # Initialize our classifier-free guidance model
    model = EDM_CFG(
        device=rank,
        in_node_nf=args.in_node_nf,
        hidden_nf=wandb.config.hidden_nf,
        ffn_embedding_dim=wandb.config.ffn_embedding_dim,
        attention_heads=wandb.config.attention_heads,
        n_layers=wandb.config.n_layers,
        tanh=wandb.config.tanh,
        coords_range=wandb.config.coords_range,
        dropout=wandb.config.dropout,
        activation_dropout=wandb.config.activation_dropout,
        backbone=backbone,
        guidance_scale=wandb.config.guidance_scale,
        condition_drop_prob=wandb.config.condition_drop_prob
    )
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        print(f'resume from {args.resume}')
    model = model.to(rank)

    train_dataset = PROTACDataset(data_path=args.data_path, prefix=args.train_data_prefix)
    val_dataset = PROTACDataset(data_path=args.data_path, prefix=args.val_data_prefix)
    test_dataset = PROTACDataset(data_path=args.data_path, prefix=args.test_data_prefix)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-12,
                                  amsgrad=True)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate)
    # test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    trainer = CFGTrainer(
        model=model,
        device=rank,
        epochs=args.epochs,
        analyze_epochs=args.analyze_epochs,
        optimizer=optimizer,
        run=run,
        loss_type=args.diffusion_loss_type,
        save_path=args.checkpoints,
        save_prefix=args.exp_name,
        guidance_scale=args.guidance_scale,  # Use the guidance scale from args
    )
    print("Start training ...")
    start_time = datetime.datetime.now()
    trainer.train(train_loader, val_loader)
    end_time = datetime.datetime.now()
    print(f'Training takes {(end_time - start_time).seconds / 60} min')


if __name__ == "__main__":
    disable_rdkit_logging()
    args = create_argparser().parse_args()
    Path(args.checkpoints).mkdir(exist_ok=True)
    world_size = torch.cuda.device_count()
    print(f'number of gpus: {world_size}')
    wandb.setup()
    main(args)
    wandb.finish()