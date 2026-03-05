import torch
import torch.nn as nn
import numpy as np
from model_base.graphormer_3d import Graphormer3D
from model_mc_v1.conditions import TimestepEmbedder, ClusterContinuousEmbedder, CategoricalEmbedder
from model_mc_v3.gated_fusion import FiLMGatedMultiConditionFusion as GatedTimeFuser

class DynamicsEfficient(nn.Module):
    def __init__(
            self,
            n_dims=3,
            in_node_nf=9,
            numerical_context_node_nf=7,
            categorical_context_node_nf=1,
            hidden_nf=128,
            context_nf=32,
            device='cpu',
            n_layers=6,
            condition_time=True,
            tanh=False,
            norm_constant=1e-6,
            normalization_factor=100,
            aggregation_method='sum',
            model='graphormer',
            ffn_embedding_dim=3072,
            attention_heads=8,
            coords_range=10,
            dropout=0.1,
            activation_dropout=0.1,
            condition_drop=0.1,
            num_category=[1],
            backbone=None,
            guidance_scale=3.0  # Default guidance scale for classifier-free guidance
    ):
        super().__init__()
        self.n_dims = n_dims
        self.numerical_context_node_nf = numerical_context_node_nf
        self.categorical_context_node_nf = categorical_context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.device = device
        self.guidance_scale = guidance_scale

        # Original backbone input dimensions
        self.backbone_input_dim = in_node_nf + 1 + condition_time  # Original features + time

        # Feature projection layer to adapt our features to the backbone's input
        self.feature_projection = nn.Linear(in_node_nf + context_nf, self.backbone_input_dim)

        # Outputs from the backbone will be processed by a second network
        if self.model == 'graphormer':
            self.dynamics = Graphormer3D(
                in_node_nf=in_node_nf + context_nf + 1 + condition_time,  # Backbone output + context + conditions
                hidden_nf=hidden_nf,
                device=device,
                n_layers=n_layers,
                tanh=tanh,
                norm_constant=norm_constant,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                ffn_embedding_dim=ffn_embedding_dim,
                attention_heads=attention_heads,
                coords_range=coords_range,
                dropout=dropout,
                activation_dropout=activation_dropout,
            )
        else:
            raise NotImplementedError

        self.edge_cache = {}

        self.context_nf = context_nf
        self.time_embedder = TimestepEmbedder(context_nf)
        self.numerical_embedder = nn.ModuleList()
        for i in range(numerical_context_node_nf):
            self.numerical_embedder.append(ClusterContinuousEmbedder(1, context_nf, condition_drop))
        self.categorical_embedder = nn.ModuleList()

        assert categorical_context_node_nf == len(num_category)
        for i in range(categorical_context_node_nf):
            self.categorical_embedder.append(CategoricalEmbedder(num_category[i], context_nf, condition_drop))
        # self.gated_fuser = GatedTimeFuser(context_nf, context_nf, context_nf)
        self.backbone = backbone
        # Freeze backbone parameters for efficiency
        # if backbone is not None:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

    def forward(self, t, xh, node_mask, linker_mask, edge_mask,
                categorical_context=None, numerical_context=None, force_unconditional=True):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B, N, N)
        - context: (B, N, C)
        - force_unconditional: Boolean to force unconditional generation (for classifier-free guidance)
        """

        bs, n_nodes = xh.shape[0], xh.shape[1]

        edges = self.get_edges(n_nodes, bs)  # (2, B*N)
        node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)
        linker_mask = linker_mask.view(bs * n_nodes, 1)  # (B*N, 1)

        # Reshaping node features & adding time feature
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)

        # Initialize context embeddings
        c = [torch.zeros((bs * n_nodes, self.context_nf)).to(self.device)]

        # Add time embedding
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)

            # Add time embedding to context
            c.append(self.time_embedder(h_time))
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)

        if categorical_context[0] is not None:
            context = categorical_context[0].view(bs * n_nodes, 1)
            h = torch.cat([h, context], dim=1)
        # Add numerical and categorical context embeddings if not forcing unconditional
        if not force_unconditional:
            # if categorical_context is not None:
            #     assert len(categorical_context) == self.categorical_context_node_nf
            #     c += self.embed_categorical_context(categorical_context, bs, n_nodes)

            if numerical_context is not None:
                assert len(numerical_context) == self.numerical_context_node_nf
                c.append(self.embed_numerical_context(numerical_context, bs, n_nodes))
        # if self.gated_fuser is not None:
        #     c = self.gated_fuser(c[1:], c[0])
        # else:
        c = torch.sum(torch.stack(c, dim=-1), dim=-1)
        # Create input for the backbone
        if self.backbone is not None:

            # Use the backbone
            size = (bs, n_nodes)
            backbone_h, backbone_x = self.backbone(
                h,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                size=size,
            )
            # Combine backbone outputs with context for final dynamics
            h_with_context = torch.cat([backbone_h, c], dim=1)

            # Final dynamics layer
            size = (bs, n_nodes)
            h_final, x_final = self.dynamics(
                h_with_context,
                backbone_x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                size=size,
            )
            vel = (x_final - x) * node_mask
        else:
            # If no backbone, use only the dynamics
            h_with_context = torch.cat([h, c], dim=1)

            size = (bs, n_nodes)
            h_final, x_final = self.dynamics(
                h_with_context,
                x,
                edges,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                size=size,
            )
            vel = (x_final - x) * node_mask

        # Remove context dimensions from final output
        if categorical_context is not None or numerical_context is not None:
            h_final = h_final[:, :-(self.context_nf + 2)]

        vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
        h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)

        return torch.cat([vel, h_final], dim=2)

    def classifier_free_guidance_forward(self, t, xh, node_mask, linker_mask, edge_mask,
                                         categorical_context=None, numerical_context=None):
        """
        Apply classifier-free guidance by running both conditional and unconditional forward passes
        and combining the results.
        """
        # Unconditional forward pass
        unconditional_out = self.forward(
            t=t,
            xh=xh,
            node_mask=node_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            categorical_context=categorical_context,
            numerical_context=numerical_context,
            force_unconditional=True
        )

        # Conditional forward pass
        conditional_out = self.forward(
            t=t,
            xh=xh,
            node_mask=node_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            categorical_context=categorical_context,
            numerical_context=numerical_context,
            force_unconditional=False
        )

        # Apply guidance
        guided_out = unconditional_out + self.guidance_scale * (conditional_out - unconditional_out)

        return guided_out

    def get_edges(self, n_nodes, batch_size):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)

    def get_force_drop_id(self, y):
        force_drop_id = torch.zeros_like(y.sum(-1))
        force_drop_id[torch.isnan(y.sum(-1))] = 1
        return force_drop_id

    def embed_categorical_context(self, context, bs, n_nodes):
        c = torch.zeros((bs * n_nodes, self.context_nf)).to(self.device)
        for i, context_item in enumerate(context):
            context_item = context_item.view(bs * n_nodes, 1)
            force_drop_id = self.get_force_drop_id(context_item)
            context_emb = self.categorical_embedder[i](context_item, self.training, force_drop_id)
            c += context_emb
        return c

    def embed_numerical_context(self, context, bs, n_nodes):
        c = torch.zeros((bs * n_nodes, self.context_nf)).to(self.device)
        for i, context_item in enumerate(context):
            # if i not in [0, 5]:
            #     continue
            context_item = context_item.expand(n_nodes, context_item.size(0), 1).reshape(bs * n_nodes, 1)
            force_drop_id = self.get_force_drop_id(context_item)
            context_emb = self.numerical_embedder[i](context_item, self.training)
            c += context_emb
        return c