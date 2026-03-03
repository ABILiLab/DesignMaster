import os
import torch 
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import remove_partial_mean_with_mask, assert_partial_mean_zero_with_mask
from utils.visualizer import save_xyz_file
from utils.dataset import create_templates_for_linker_generation

from tqdm import tqdm


class StageAwareTrainer:
    def __init__(
        self,
        model,
        device,
        epochs,
        analyze_epochs,
        optimizer,
        run,
        loss_type,
        save_path,
        save_prefix,
        n_stability_samples=10,
        # Stage-aware specific parameters
        condition_analysis_path="condition_analysis",
    ) -> None:
        self.device = device
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.run = run
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.loss_type = loss_type
        self.analyze_epochs = analyze_epochs
        self.n_stability_samples = n_stability_samples
        self.condition_analysis_path = condition_analysis_path
        
        # Create the condition analysis directory
        os.makedirs(condition_analysis_path, exist_ok=True)
    
    def pred(self, dataloader, output_dir, sample_fn=None, delta_linker_size=0):
        if delta_linker_size != 0:
            linker_path = 'linker_' + str(delta_linker_size)
            output_dir = os.path.join(output_dir, linker_path)
        
        # Initialize condition usage tracking
        all_condition_stats = []
        
        for data in tqdm(dataloader):
            uuids = []
            true_names = []
            frag_names = []
            for uuid in data['uuid']:
                uuid = str(uuid)
                uuids.append(uuid)
                true_names.append(f'{uuid}/true')
                frag_names.append(f'{uuid}/frag')
                os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

            # Removing COM of fragment from the atom coordinates
            h, x, node_mask, frag_mask = data['one_hot'], data['positions'], data['atom_mask'], data['fragment_mask']
            
            center_of_mass_mask = data['fragment_mask']
            x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
            assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

            # Saving ground-truth molecules
            save_xyz_file(output_dir, h, x, node_mask, true_names)

            # Saving fragments
            save_xyz_file(output_dir, h, x, frag_mask, frag_names)

            # Sampling and saving generated molecules
            for i in range(self.n_stability_samples):
                chain, node_mask = self.sample_chain(data, sample_fn, keep_frames=1, delta_linker_size=delta_linker_size)
                x = chain[0][:, :, :3]
                h = chain[0][:, :, 3:]
                
                pred_names = [f'{uuid}/{i}' for uuid in uuids]
                save_xyz_file(output_dir, h, x, node_mask, pred_names)
                
                # Collect condition usage statistics for this sample
                if hasattr(self.model, 'get_condition_usage_stats'):
                    stats = self.model.get_condition_usage_stats()
                    # Add metadata
                    for j, uuid in enumerate(uuids):
                        stats_copy = stats.copy()
                        stats_copy["uuid"] = uuid
                        stats_copy["sample_id"] = i
                        all_condition_stats.append(stats_copy)
        
        # Save condition analysis
        if all_condition_stats:
            # Aggregate statistics across all molecules
            aggregated_stats = self.aggregate_condition_stats(all_condition_stats)
            
            # Save detailed and aggregated statistics
            analysis_file = os.path.join(self.condition_analysis_path, f"{os.path.basename(output_dir)}_condition_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump({
                    "detailed": all_condition_stats,
                    "aggregated": aggregated_stats
                }, f, indent=2)
            
            # Generate visualizations
            self.visualize_condition_usage(aggregated_stats, os.path.join(self.condition_analysis_path, f"{os.path.basename(output_dir)}_condition_usage"))
    
    def aggregate_condition_stats(self, all_stats):
        """Aggregate condition statistics across all molecules"""
        if not all_stats:
            return {}
            
        # Initialize aggregation structure
        first_stat = all_stats[0]
        aggregated = {
            "total_steps": 0,
            "stage_distribution": {stage: 0 for stage in first_stat["stage_distribution"].keys()},
            "condition_usage": {
                cond: {stage: 0 for stage in first_stat["condition_usage"][cond].keys()}
                for cond in first_stat["condition_usage"].keys()
            }
        }
        
        # Sum up statistics
        for stat in all_stats:
            aggregated["total_steps"] += stat["total_steps"]
            
            for stage, value in stat["stage_distribution"].items():
                aggregated["stage_distribution"][stage] += value * stat["total_steps"] / 100
                
            for cond, stages in stat["condition_usage"].items():
                for stage, value in stages.items():
                    aggregated["condition_usage"][cond][stage] += value * stat["total_steps"] / 100
        
        # Convert to percentages
        for stage in aggregated["stage_distribution"].keys():
            aggregated["stage_distribution"][stage] = (
                aggregated["stage_distribution"][stage] / aggregated["total_steps"] * 100
            )
            
        for cond in aggregated["condition_usage"].keys():
            for stage in aggregated["condition_usage"][cond].keys():
                if stage != "overall":  # Skip overall which is calculated differently
                    stage_steps = sum(stat["stage_distribution"][stage] * stat["total_steps"] / 100 
                                     for stat in all_stats)
                    if stage_steps > 0:
                        aggregated["condition_usage"][cond][stage] = (
                            aggregated["condition_usage"][cond][stage] / stage_steps * 100
                        )
            
            # Calculate overall percentage
            aggregated["condition_usage"][cond]["overall"] = (
                sum(aggregated["condition_usage"][cond][s] * aggregated["stage_distribution"][s] / 100
                   for s in ["early", "mid", "late"])
            )
        
        return aggregated
    
    def visualize_condition_usage(self, stats, output_prefix):
        """Create visualizations of condition usage statistics"""
        # 1. Stage distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        stages = list(stats["stage_distribution"].keys())
        values = [stats["stage_distribution"][s] for s in stages]
        ax.pie(values, labels=stages, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Distribution of Diffusion Stages')
        plt.savefig(f"{output_prefix}_stages.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Condition usage by stage
        conditions = list(stats["condition_usage"].keys())
        
        # Overall usage
        fig, ax = plt.subplots(figsize=(10, 6))
        overall_values = [stats["condition_usage"][cond]["overall"] for cond in conditions]
        y_pos = np.arange(len(conditions))
        ax.barh(y_pos, overall_values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(conditions)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Usage Percentage')
        ax.set_title('Overall Condition Usage')
        plt.savefig(f"{output_prefix}_overall.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-stage usage
        for stage in ["early", "mid", "late"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            stage_values = [stats["condition_usage"][cond][stage] for cond in conditions]
            ax.barh(y_pos, stage_values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(conditions)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Usage Percentage')
            ax.set_title(f'Condition Usage in {stage.capitalize()} Stage')
            plt.savefig(f"{output_prefix}_{stage}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Heatmap of condition usage across stages
        fig, ax = plt.subplots(figsize=(10, 8))
        data = np.array([[stats["condition_usage"][cond][stage] for stage in ["early", "mid", "late"]]
                         for cond in conditions])
        im = ax.imshow(data, cmap='YlOrRd')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Usage Percentage', rotation=-90, va="bottom")
        
        # Show all ticks and label them
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_xticklabels(["Early", "Mid", "Late"])
        ax.set_yticklabels(conditions)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(conditions)):
            for j in range(3):
                text = ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center", color="black")
        
        ax.set_title("Condition Usage Heatmap by Diffusion Stage")
        fig.tight_layout()
        plt.savefig(f"{output_prefix}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.train_epoch(train_loader)
            self.val_epoch(val_loader, epoch)

    def train_epoch(self, loader):
        self.model.train()
        step_outputs = []
        for data in loader:
            self.optimizer.zero_grad()
            output = self._step(data)
            output['loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            for metric in output.keys():
                self.run.log({f'{metric}/train_step': output[metric]})
            step_outputs.append(output)
        with torch.no_grad():
            for metric in step_outputs[0].keys():
                avg_metric = StageAwareTrainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/train_epoch': avg_metric})
    
    def val_epoch(self, loader, epoch):
        best_loss = float('inf')
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = StageAwareTrainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/val': avg_metric})
                if metric == 'loss' and avg_metric < best_loss:
                    best_loss = avg_metric
                    torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_best.ckpt')
                print(f'{metric}/val: {avg_metric}')
            
            if (epoch + 1) % self.analyze_epochs == 0:
                # Save checkpoint
                torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_{epoch}.ckpt')
                
                # Generate and analyze samples periodically
                if hasattr(self.model, 'get_condition_usage_stats'):
                    # Sample a small validation batch for condition analysis
                    val_data = next(iter(loader))
                    self.analyze_condition_usage(val_data, epoch)

    def analyze_condition_usage(self, data, epoch):
        """Analyze condition usage on a validation batch and save results"""
        # Generate samples
        chain, node_mask = self.sample_chain(data, keep_frames=None)
        
        # Get condition usage statistics
        if hasattr(self.model, 'get_condition_usage_stats'):
            stats = self.model.get_condition_usage_stats()
            
            # Save statistics
            os.makedirs(os.path.join(self.condition_analysis_path, f"epoch_{epoch}"), exist_ok=True)
            with open(os.path.join(self.condition_analysis_path, f"epoch_{epoch}/condition_usage.json"), 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations
            self.visualize_condition_usage(stats, os.path.join(self.condition_analysis_path, f"epoch_{epoch}/condition_usage"))
            
            # Log to tracking system if available
            if self.run is not None:
                # Log overall condition usage percentages
                for cond, stages in stats["condition_usage"].items():
                    self.run.log({f'condition/{cond}/overall': stages["overall"]})
                
                # Log stage distribution
                for stage, pct in stats["stage_distribution"].items():
                    self.run.log({f'stage/{stage}': pct})
    
    def test_epoch(self, loader):
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = StageAwareTrainer.aggregate_metric(step_outputs, metric)
                print(f'{metric}/test: {avg_metric}')
                
            # Generate predictions with condition analysis
            self.pred(loader, f'{self.save_prefix}_test')
    
    def _step(self, data):
        l2_loss = self.model(data)
        if self.loss_type == 'l2':
            loss, loss_x, loss_h = l2_loss
        else:
            raise NotImplementedError(self.loss_type)
        
        metrics = {
            'loss': loss,
            'loss_x': loss_x,
            'loss_h': loss_h,
        }

        return metrics

    def sample_chain(self, data, sample_fn=None, keep_frames=None, delta_linker_size=0):
        if sample_fn is None:
            linker_sizes = data['linker_mask'].sum(1).view(-1).int() + delta_linker_size
        else:
            linker_sizes = sample_fn(data)

        template_data = create_templates_for_linker_generation(data, linker_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']
        context = fragment_mask
        center_of_mass_mask = fragment_mask

        molecular_weight = torch.tensor(data['molecular_weight'], dtype=torch.float).unsqueeze(1).to(self.device)
        h_bond_acceptor = torch.tensor(data['H_bond_acceptor_cnt'], dtype=torch.float).unsqueeze(1).to(self.device)
        h_bond_donor = torch.tensor(data['H_bond_donor_cnt'], dtype=torch.float).unsqueeze(1).to(self.device)
        log_p = torch.tensor(data['XLogP3'], dtype=torch.float).unsqueeze(1).to(self.device)
        rotatable_bond_count = torch.tensor(data['rotatable_bond_cnt'], dtype=torch.float).unsqueeze(1).to(self.device)
        heavy_atom_count = torch.tensor(data['heavy_atom_cnt'], dtype=torch.float).unsqueeze(1).to(self.device)
        topological_polar_surface_area = torch.tensor(data['topological_polar_surface_area'],
                                                      dtype=torch.float).unsqueeze(1).to(self.device)
        numerical_context = [molecular_weight, h_bond_acceptor, h_bond_donor, log_p, rotatable_bond_count,
                      heavy_atom_count, topological_polar_surface_area]

        x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        chain = self.model.sample_chain(
            x=x.to(self.device),
            h=h.to(self.device),
            node_mask=node_mask.to(self.device),
            edge_mask=edge_mask.to(self.device),
            fragment_mask=fragment_mask.to(self.device),
            linker_mask=linker_mask.to(self.device),
            context=context.to(self.device),
            numerical_context=numerical_context,
            keep_frames=keep_frames
        )
        return chain, node_mask

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        """Aggregate metrics across multiple steps"""
        return torch.tensor([out[metric] for out in step_outputs]).mean()