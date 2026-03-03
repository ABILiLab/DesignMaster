import os
import torch
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import distributed, DataLoader
from pathlib import Path

from model_mc_v3.edm_mc_v3 import EDM_CFG  # Import the new model
from utils.utils import add_dict_to_argparser
from utils.dataset import PROTACDataset, collate
from trainer.trainer_mc_v3 import CFGTrainer  # Import the new trainer


def create_argparser():
    defaults = dict(
        model_path="checkpoints/protacs_mc_cfg_best.ckpt",
        output_dir="protacs_mc_case-3",
        exp_name='',
        data_path="datasets-3.0/datasets",
        test_data_prefix="protacs_test_cases",
        # data_path="dataset-case",
        # test_data_prefix="test-case-2",
        checkpoints='checkpoints',
        n_samples=100,
        batch_size=600,
        epochs=None,
        in_node_nf=9,
        num_workers=6,
        linker_size=0,
        use_backbone=True
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
        dropout=0,
        activation_dropout=0,
        # Classifier-free guidance specific parameters
        guidance_scale=None,  # Default guidance scale for testing
        condition_drop_prob=0.1,  # Not used during testing but needed for model definition
        # Allow testing with multiple guidance values
        guidance_scales=[1.0, 3.0, 5.0, 7.0],  # Test with different guidance values
        run_all_guidance=False,  # If True, will run with all guidance scales defined above
    )
    parser = argparse.ArgumentParser()
    defaults.update(egnn_defaults)
    add_dict_to_argparser(parser, defaults)
    return parser


def main(args):
    # Load the pre-trained backbone model
    from model_base.edm import EDM as EDM_backbone
    rank = 0
    
    # Initialize the backbone if needed
    backbone = None
    if hasattr(args, 'use_backbone') and args.use_backbone:
        backbone_model = EDM_backbone(
            device=rank,
            in_node_nf=args.in_node_nf,
            hidden_nf=args.hidden_nf,
            ffn_embedding_dim=args.ffn_embedding_dim,
            attention_heads=args.attention_heads,
            n_layers=args.n_layers,
            tanh=args.tanh,
            coords_range=args.coords_range,
            dropout=args.dropout,
            activation_dropout=args.activation_dropout
        )
        backbone_model.load_state_dict(torch.load('checkpoints/geom_best.ckpt'))
        backbone = backbone_model.dynamics.dynamics
    
    # Initialize our classifier-free guidance model
    model = EDM_CFG(
        device=rank,
        in_node_nf=args.in_node_nf,
        hidden_nf=args.hidden_nf,
        ffn_embedding_dim=args.ffn_embedding_dim,
        attention_heads=args.attention_heads,
        n_layers=args.n_layers,
        tanh=args.tanh,
        coords_range=args.coords_range,
        dropout=args.dropout,
        activation_dropout=args.activation_dropout,
        backbone=backbone,
        guidance_scale=args.guidance_scale,
        condition_drop_prob=args.condition_drop_prob
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(rank)
    model.eval()
    
    # Load test dataset
    test_dataset = PROTACDataset(data_path=args.data_path, prefix=args.test_data_prefix)
    print(test_dataset)
    dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate)

    # Initialize the trainer
    trainer = CFGTrainer(
        model=model,
        device=rank,
        epochs=args.epochs,
        analyze_epochs=None,
        n_stability_samples=args.n_samples,
        optimizer=None,
        run=None,
        loss_type=args.diffusion_loss_type,
        save_path=args.checkpoints,
        save_prefix=args.exp_name,
        guidance_scale=args.guidance_scale,  # Default guidance scale
        property_adjustment=True,
    )

    # Define the sample function for linker size if needed
    sample_fn = None
    if args.linker_size != 0:
        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=rank, dtype=torch.int) * args.linker_size
    # if args.linker_size != 0:
    #     def sample_fn(_data):
    #         pos = _data['positions']
    #         batch_size, pos_dim = pos.shape
    #         total_dim = args.linker_size + pos_dim
    #         return torch.ones(batch_size, total_dim, device=rank, dtype=torch.int)


    # Run prediction
    if args.run_all_guidance:
        # Test with multiple guidance values
        for gs in args.guidance_scales:
            print(f"\nTesting with guidance scale: {gs}")
            output_dir = f"{args.output_dir}_gs_{gs}"
            Path(output_dir).mkdir(exist_ok=True)
            trainer.pred(dataloader, output_dir, sample_fn, guidance_scale=gs)
    else:
        # Test with the single specified guidance scale
        print(f"\nTesting with guidance scale: {args.guidance_scale}")
        Path(args.output_dir).mkdir(exist_ok=True)
        trainer.pred(dataloader, args.output_dir, sample_fn, save_chains_for_viz=True)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    world_size = torch.cuda.device_count()
    print(f'number of gpus: {world_size}')
    main(args)