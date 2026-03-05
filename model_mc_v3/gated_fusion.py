import torch
import torch.nn as nn


class FiLMGatedMultiConditionFusion(nn.Module):
    def __init__(self, input_dim, output_dim, timestep_emb_dim, num_conditions=7):
        super().__init__()
        self.num_conditions = num_conditions
        self.output_dim = output_dim

        # Condition-specific processors
        self.condition_processors = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_conditions)
        ])
        
        # FiLM parameters for each condition
        self.gamma_layers = nn.ModuleList([
            nn.Linear(timestep_emb_dim, output_dim) for _ in range(self.num_conditions)
        ])
        self.beta_layers = nn.ModuleList([
            nn.Linear(timestep_emb_dim, output_dim) for _ in range(self.num_conditions)
        ])
        
        # Gate networks for each condition
        self.gate_layers = nn.ModuleList([
            nn.Linear(timestep_emb_dim, 1) for _ in range(self.num_conditions)
        ])
        
        # Final projection
        self.output_projection = nn.Linear(output_dim, output_dim)
    
    def forward(self, conditions, t_emb):

        # Process each condition with FiLM and gating
        processed_conditions = []
        for i, condition in enumerate(conditions):
            # Basic processing to common dimension
            processed = self.condition_processors[i](condition)
            
            # Apply FiLM modulation
            gamma = self.gamma_layers[i](t_emb)
            beta = self.beta_layers[i](t_emb)
            modulated = gamma * processed + beta
            
            # Apply gate to control importance
            gate = torch.sigmoid(self.gate_layers[i](t_emb))
            gated = modulated * gate
            
            processed_conditions.append(gated)
        
        # Sum the modulated and gated conditions
        fused_condition = sum(processed_conditions)
        
        # Final projection to ensure output quality
        output = self.output_projection(fused_condition)
        
        return output

    def get_importance_weights(self, t, t_embedder):
        """Helper method to extract gate values for interpretation"""
        t_emb = t_embedder(t.view(-1, 1))
        gates = [torch.sigmoid(gate_layer(t_emb)) for gate_layer in self.gate_layers]
        return gates
