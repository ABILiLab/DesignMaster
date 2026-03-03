import os
import torch
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import distributed, DataLoader
from pathlib import Path

from model_base.edm import EDM
from utils.utils import add_dict_to_argparser
from utils.dataset import PROTACDataset, collate
from utils.const import NUMBER_OF_ATOM_TYPES
from trainer.trainer import Trainer


def create_argparser():
    defaults = dict(
        model_path="checkpoints/baseline_best.ckpt",
        output_dir="protacs_baseline_case-3",
        exp_name='',
        data_path="datasets-3.0/datasets",
        test_data_prefix="protacs_test_cases",
        # data_path="datasets-3.0/datasets",
        # test_data_prefix="protacs_test",
        checkpoints='checkpoints',
        n_samples=100,
        batch_size=600,
        epochs=None,
        in_node_nf=9,
        num_workers=6,
        linker_size=0,
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
    )
    parser = argparse.ArgumentParser()
    defaults.update(egnn_defaults)
    add_dict_to_argparser(parser, defaults)
    return parser

def main(args):
    # rank = hvd.local_rank()
    rank = 0
    model = EDM(
        device=rank,
        in_node_nf=args.in_node_nf,
        hidden_nf=args.hidden_nf,
        ffn_embedding_dim=args.ffn_embedding_dim,
        attention_heads=args.attention_heads,
        n_layers=args.n_layers,
        tanh=args.tanh,
        coords_range=args.coords_range,
        dropout=args.dropout,
        activation_dropout= args.activation_dropout,
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(rank)
    model.eval()
    test_dataset = PROTACDataset(data_path=args.data_path, prefix=args.test_data_prefix)
    dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    # dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank()), collate_fn=collate)

    trainer = Trainer(
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
    )
    
    sample_fn = None
    if args.linker_size > 0:
        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=rank, dtype=torch.int) * args.linker_size
    # if args.linker_size != 0:
    #     def sample_fn(_data):
    #         pos = _data['positions']
    #         batch_size = pos.shape[0]
    #         total_dim = args.linker_size + int(_data['linker_mask'].sum().item())
    #         print(total_dim)
    #         return torch.ones(batch_size, total_dim, device=rank, dtype=torch.int)
    
    trainer.pred(dataloader, args.output_dir, sample_fn)

if __name__ == "__main__":

    args = create_argparser().parse_args()
    Path(args.output_dir).mkdir(exist_ok=True)
    world_size = torch.cuda.device_count()
    print(f'number of gpus: {world_size}')
    # mp.set_sharing_strategy('file_system')
    # hvd.init()
    main(args)
