import os
import torch
from utils.utils import remove_partial_mean_with_mask, assert_partial_mean_zero_with_mask
from utils.visualizer import save_xyz_file
from utils.dataset import create_templates_for_linker_generation
import pandas as pd
import joblib
from tqdm import tqdm


class CFGTrainer:
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
            guidance_scale=None,  # Default guidance scale for sampling
            property_adjustment=True,
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
        self.guidance_scale = guidance_scale  # Store guidance scale for sampling
        self.property_adjustment = property_adjustment

    def pred(self, dataloader, output_dir, sample_fn=None, guidance_scale=None, 
         save_chains_for_viz=False, max_viz_samples=5, keep_frames_for_viz=500):
        """
        Generate predictions with optional guidance scale control and chain visualization
        
        Args:
            dataloader: DataLoader for test data
            output_dir: Directory to save outputs
            sample_fn: Optional sampling function
            guidance_scale: Guidance scale for generation
            save_chains_for_viz: Whether to save full chains for visualization
            max_viz_samples: Maximum number of samples to save chains for
            keep_frames_for_viz: Number of frames to keep in chain for visualization
        """
        # Use provided guidance scale or default
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        viz_sample_count = 0  # Counter for visualization samples
        
        for batch_idx, data in enumerate(tqdm(dataloader)):
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
            
            # Determine if we should save chains for this batch
            save_chains_this_batch = (save_chains_for_viz and 
                                    viz_sample_count < max_viz_samples)
            
            # Sampling and saving generated molecules
            for i in range(self.n_stability_samples):
                # Decide whether to keep full chain for this specific sample
                if save_chains_this_batch and i == 0:  # Only save chain for first sample of selected batches
                    chain, node_mask = self.sample_chain(
                        data,
                        sample_fn,
                        keep_frames=keep_frames_for_viz,  # Keep more frames for visualization
                        guidance_scale=guidance_scale
                    )
                    # Save full chain for visualization
                    self.save_chain_for_visualization(output_dir, chain, node_mask, uuids, batch_idx)
                    viz_sample_count += 1
                else:
                    # Regular sampling with minimal frames
                    chain, node_mask = self.sample_chain(
                        data,
                        sample_fn,
                        keep_frames=1,
                        guidance_scale=guidance_scale
                    )
                # Extract final frame for regular prediction saving
                x = chain[0][:, :, :3]  
                h = chain[0][:, :, 3:]  
                pred_names = [f'{uuid}/{i}' for uuid in uuids]
                save_xyz_file(output_dir, h, x, node_mask, pred_names)

    def save_chain_for_visualization(self, output_dir, chain, node_mask, uuids, batch_idx):
        """
        Save full chain data for visualization purposes
        
        Args:
            output_dir: Base output directory
            chain: Full chain tensor [n_frames, batch_size, n_atoms, features]
            node_mask: Node mask for valid atoms
            uuids: List of UUIDs for this batch
            batch_idx: Batch index for unique naming
        """
        import pickle
        import numpy as np
        
        # Create visualization subdirectory
        viz_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert chain to numpy and save
        chain_np = chain.cpu().numpy() if hasattr(chain, 'cpu') else np.array(chain)
        node_mask_np = node_mask.cpu().numpy() if hasattr(node_mask, 'cpu') else np.array(node_mask)
        
        for mol_idx, uuid in enumerate(uuids):
            # Extract this molecule's chain
            mol_chain = chain_np[:, mol_idx, :, :]  # [n_frames, n_atoms, features]
            mol_mask = node_mask_np[mol_idx, :]     # [n_atoms]
            
            # Save chain data
            chain_data = {
                'chain': mol_chain,
                'node_mask': mol_mask,
                'uuid': uuid,
                'n_frames': mol_chain.shape[0],
                'n_atoms': mol_chain.shape[1]
            }
            
            chain_file = os.path.join(viz_dir, f'{uuid}_chain_batch{batch_idx}.pkl')
            with open(chain_file, 'wb') as f:
                pickle.dump(chain_data, f)
            
            # Also save as separate XYZ files for each frame (optional)
            if False:  # Set to False if you don't want individual frame files
                frame_dir = os.path.join(viz_dir, f'{uuid}_frames_batch{batch_idx}')
                os.makedirs(frame_dir, exist_ok=True)
                
                for frame_idx in range(mol_chain.shape[0]):
                    frame_x = mol_chain[frame_idx, :, :3]  # positions
                    frame_h = mol_chain[frame_idx, :, 3:]  # atom types
                    frame_name = [f'frame_{frame_idx:03d}']
                    
                    # Expand dimensions to match save_xyz_file expected format
                    # frame_h_batch = np.expand_dims(frame_h, 0)
                    # frame_x_batch = np.expand_dims(frame_x, 0)
                    # frame_mask_batch = np.expand_dims(mol_mask, 0)

                    save_xyz_file(frame_dir, frame_h, frame_x, mol_mask, frame_name)

    def train(self, train_loader, val_loader):
        """Train the model for the specified number of epochs"""
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.train_epoch(train_loader)
            self.val_epoch(val_loader, epoch)

    def train_epoch(self, loader):
        """Train for one epoch"""
        self.model.train()
        step_outputs = []
        for data in loader:
            self.optimizer.zero_grad()
            output = self._step(data)
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            for metric in output.keys():
                self.run.log({f'{metric}/train_step': output[metric]})
            step_outputs.append(output)
        with torch.no_grad():
            for metric in step_outputs[0].keys():
                avg_metric = CFGTrainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/train_epoch': avg_metric})

    def val_epoch(self, loader, epoch):
        """Validate the model and save checkpoints"""
        best_loss = float('inf')
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = CFGTrainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/val': avg_metric})
                if metric == 'loss' and avg_metric < best_loss:
                    best_loss = avg_metric
                    torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_best.ckpt')
                print(f'{metric}/val: {avg_metric}')

            if (epoch + 1) % self.analyze_epochs == 0:
                torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_{epoch}.ckpt')

                # Generate validation samples with different guidance scales
                # This helps monitor how different guidance values affect generation
                if hasattr(self.model, 'sample_chain'):
                    val_sample_dir = f'{self.save_path}/samples_epoch_{epoch}'
                    os.makedirs(val_sample_dir, exist_ok=True)

                    # Sample a small subset of validation data
                    sample_data = next(iter(loader))
                    for gs in [1.0, 3.0, 5.0]:  # Test different guidance scales
                        gs_dir = f"{val_sample_dir}/gs_{gs}"
                        self.sample_for_visualization(sample_data, gs_dir, guidance_scale=gs)

    def sample_for_visualization(self, data, output_dir, guidance_scale=None):
        """Sample a few molecules for visualization"""
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            chain, node_mask = self.sample_chain(
                data,
                keep_frames=1,
                guidance_scale=guidance_scale
            )
            x = chain[0][:, :, :3]
            h = chain[0][:, :, 3:]

            # Sample only a few molecules to save time
            num_samples = min(5, x.shape[0])
            x = x[:num_samples]
            h = h[:num_samples]
            node_mask = node_mask[:num_samples]

            pred_names = [f'sample_{i}' for i in range(num_samples)]
            save_xyz_file(output_dir, h, x, node_mask, pred_names)

    def test_epoch(self, loader, guidance_scale=None):
        """Test the model and generate predictions"""
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = CFGTrainer.aggregate_metric(step_outputs, metric)
                print(f'{metric}/test: {avg_metric}')

            # Generate with the specified guidance scale
            test_output_dir = f'{self.save_prefix}_test_gs_{guidance_scale}'
            self.pred(loader, test_output_dir, guidance_scale=guidance_scale)

    def _step(self, data):
        """Perform a single training/validation step"""
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

    def sample_chain(self, data, sample_fn=None, keep_frames=None, guidance_scale=None):
        """
        Sample a chain with optional guidance scale control
        """
        # Use provided guidance scale or default
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        if sample_fn is None:
            linker_sizes = data['linker_mask'].sum(1).view(-1).int()
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

        # Prepare numerical properties as conditioning
        # Original properties from data
        orig_properties = {
            'molecular_weight': data['molecular_weight'],
            'H_bond_acceptor_cnt': data['H_bond_acceptor_cnt'],
            'H_bond_donor_cnt': data['H_bond_donor_cnt'],
            'XLogP3': data['XLogP3'],
            'rotatable_bond_cnt': data['rotatable_bond_cnt'],
            'heavy_atom_cnt': data['heavy_atom_cnt'],
            'topological_polar_surface_area': data['topological_polar_surface_area']
        }

        if sample_fn is not None and self.property_adjustment:
            # Get the original linker size
            orig_linker_size = data['linker_mask'].sum(1).view(-1).int()

            # For each molecule in the batch
            for i in range(len(orig_linker_size)):
                # Create pseudo-SMILES of the new size for prediction
                # This is a workaround since we don't have actual SMILES, just using a string of the right length
                new_linker_size = linker_sizes[i]

                # Load our pre-trained models for property prediction
                # These should be loaded once and stored as class attributes, not here in the function
                if not hasattr(self, 'property_models'):
                    self.load_property_prediction_models()

                # Predict new properties based on the new linker size
                new_properties = self.predict_properties_from_linker_size(new_linker_size.item())

                # Update properties for this molecule
                orig_properties['molecular_weight'][i] += new_properties['Molecular Weight']
                orig_properties['H_bond_acceptor_cnt'][i] += new_properties['Hydrogen Bond Acceptor Count']
                orig_properties['H_bond_donor_cnt'][i] += new_properties['Hydrogen Bond Donor Count']
                orig_properties['XLogP3'][i] += new_properties['XLogP3']
                orig_properties['rotatable_bond_cnt'][i] += new_properties['Rotatable Bond Count']
                orig_properties['heavy_atom_cnt'][i] += new_properties['Heavy Atom Count']
                orig_properties['topological_polar_surface_area'][i] += new_properties['Topological Polar Surface Area']

        # Convert properties to tensors for the model
        molecular_weight = torch.tensor(orig_properties['molecular_weight'], dtype=torch.float).unsqueeze(1).to(
            self.device)
        h_bond_acceptor = torch.tensor(orig_properties['H_bond_acceptor_cnt'], dtype=torch.float).unsqueeze(1).to(
            self.device)
        h_bond_donor = torch.tensor(orig_properties['H_bond_donor_cnt'], dtype=torch.float).unsqueeze(1).to(self.device)
        log_p = torch.tensor(orig_properties['XLogP3'], dtype=torch.float).unsqueeze(1).to(self.device)
        rotatable_bond_count = torch.tensor(orig_properties['rotatable_bond_cnt'], dtype=torch.float).unsqueeze(1).to(
            self.device)
        heavy_atom_count = torch.tensor(orig_properties['heavy_atom_cnt'], dtype=torch.float).unsqueeze(1).to(
            self.device)
        topological_polar_surface_area = torch.tensor(orig_properties['topological_polar_surface_area'],
                                                      dtype=torch.float).unsqueeze(1).to(self.device)
        numerical_context = [molecular_weight, h_bond_acceptor, h_bond_donor, log_p, rotatable_bond_count,
                             heavy_atom_count, topological_polar_surface_area]

        x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        # Call the model's sample_chain method with the correct parameter names
        chain = self.model.sample_chain(
            x=x.to(self.device),
            h=h.to(self.device),
            node_mask=node_mask.to(self.device),
            edge_mask=edge_mask.to(self.device),
            fragment_mask=fragment_mask.to(self.device),
            linker_mask=linker_mask.to(self.device),
            context=context.to(self.device),
            numerical_context=numerical_context,  # Updated parameter name
            keep_frames=keep_frames,
            guidance_scale=guidance_scale,  # Pass guidance scale to control conditioning strength
        )
        return chain, node_mask

    def load_property_prediction_models(self):
        """Load the pre-trained property prediction models and scalers."""
        self.property_models = {}
        self.property_scalers = {}

        property_columns = [
            'Molecular Weight', 'XLogP3', 'Heavy Atom Count', 'Ring Count',
            'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count',
            'Rotatable Bond Count', 'Topological Polar Surface Area'
        ]

        for prop in property_columns:
            model_path = f'linker2prop_res/model_{prop.replace(" ", "_").lower()}.pkl'
            scaler_path = f'linker2prop_res/scaler_{prop.replace(" ", "_").lower()}.pkl'

            self.property_models[prop] = joblib.load(model_path)
            self.property_scalers[prop] = joblib.load(scaler_path)

    def predict_properties_from_linker_size(self, linker_size):
        """Predict properties for a given linker size."""
        # Create feature vector with just the linker size
        X = pd.DataFrame([{'smiles_length': linker_size}])

        # Make predictions for each property
        predictions = {}
        for prop, model in self.property_models.items():
            scaler = self.property_scalers[prop]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            predictions[prop] = pred

        return predictions

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        """Aggregate metrics across multiple steps"""
        return torch.tensor([out[metric] for out in step_outputs]).mean()