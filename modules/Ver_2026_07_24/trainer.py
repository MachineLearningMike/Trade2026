# trainer.py
# Resumable trainer with checkpoint management and interruption handling

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import signal
import sys
import atexit
import math
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple
from config import get_config


# ═══════════════════════════════════════════════════════════
# TRAINING STATE MANAGER
# ═══════════════════════════════════════════════════════════

class TrainingState:
    """Manages saving and loading of complete training state with atomic writes."""
    
    def __init__(self, save_dir='checkpoints', save_every_n_minibatches=100, 
                 max_checkpoints=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_minibatches = save_every_n_minibatches
        self.max_checkpoints = max_checkpoints
        self.state_file = self.save_dir / 'training_state.json'
        
    def save_checkpoint(self, model, optimizer, dataloader_state, epoch, batch_idx, 
                        total_minibatches_processed, loss, is_emergency=False):
        """Save checkpoint with atomic write to prevent corruption."""
        checkpoint = {
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataloader_state': dataloader_state,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
        }
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        prefix = 'emergency' if is_emergency else 'checkpoint'
        checkpoint_file = self.save_dir / f'{prefix}_minibatch{total_minibatches_processed}_{timestamp}.pt'
        
        # Atomic save: write to temp file, then rename
        temp_file = checkpoint_file.with_suffix('.tmp')
        torch.save(checkpoint, temp_file)
        
        # Retry rename with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(temp_file, checkpoint_file)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    # If rename fails, try copy as fallback
                    try:
                        import shutil
                        shutil.copy2(temp_file, checkpoint_file)
                        os.remove(temp_file)
                    except:
                        raise e
        
        # Save state info with retry
        state_info = {
            'latest_checkpoint': str(checkpoint_file),
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
        }
        
        # Atomic state file update with retry
        temp_state = self.state_file.with_suffix('.tmp')
        with open(temp_state, 'w') as f:
            json.dump(state_info, f, indent=2)
        
        for attempt in range(max_retries):
            try:
                os.replace(temp_state, self.state_file)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    # If rename fails, write directly with a different name
                    fallback_file = self.save_dir / f'training_state_{timestamp}.json'
                    with open(fallback_file, 'w') as f:
                        json.dump(state_info, f, indent=2)
                    print(f"⚠ State file saved as fallback: {fallback_file}")
                    break
        
        self._cleanup_old_checkpoints()
        
        if is_emergency:
            print(f"\n{'!'*60}")
            print(f"EMERGENCY CHECKPOINT SAVED at minibatch {total_minibatches_processed}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'!'*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Checkpoint saved at minibatch {total_minibatches_processed}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'='*60}")
        
    def load_latest_checkpoint(self, model, optimizer, dataloader):
        """Load the latest checkpoint. Returns None if nothing to resume."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("⚠ State file corrupted. Starting fresh.")
            return None
        
        latest_checkpoint = Path(state_info['latest_checkpoint'])
        
        if not latest_checkpoint.exists():
            print(f"⚠ Checkpoint file not found. Starting fresh.")
            return None
        
        print(f"Found checkpoint: {latest_checkpoint.name}")
        
        try:
            checkpoint = torch.load(latest_checkpoint)
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}. Starting fresh.")
            return None
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        total_minibatches_processed = checkpoint['total_minibatches_processed']
        loss = checkpoint['loss']
        is_emergency = checkpoint.get('is_emergency', False)
        
        if 'dataloader_state' in checkpoint:
            dataloader_state = checkpoint['dataloader_state']
            dataloader.restore_state(dataloader_state)
            
            if is_emergency:
                print(f"⚠ Resuming from EMERGENCY checkpoint")
            else:
                print(f"Resuming from checkpoint")
            print(f"  Minibatch: {total_minibatches_processed}")
            print(f"  Epoch {epoch+1}, batch {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            
            return epoch, batch_idx, total_minibatches_processed, loss, dataloader_state, is_emergency
        
        return None
    
    def _cleanup_old_checkpoints(self):
        all_checkpoints = sorted(
            list(self.save_dir.glob('checkpoint_minibatch*.pt')) + 
            list(self.save_dir.glob('emergency_minibatch*.pt')),
            key=os.path.getmtime,
            reverse=True
        )
        
        for old_file in all_checkpoints[self.max_checkpoints:]:
            try:
                old_file.unlink()
            except:
                pass


# ═══════════════════════════════════════════════════════════
# RESUMABLE TRAINER
# ═══════════════════════════════════════════════════════════

class ResumableTrainer:
    """
    Trainer with robust handling of all interruption scenarios.
    """
    
    def __init__(self, model, loss_fn, optimizer, dataloader, 
                 save_dir='checkpoints', device='cpu',
                 save_every_n_minibatches=100):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        
        self.state_manager = TrainingState(
            save_dir=save_dir,
            save_every_n_minibatches=save_every_n_minibatches
        )
        
        # Control flags
        self.stop_requested = False
        self.is_running = False
        self.ctrl_c_count = 0
        
        # Training state
        self.total_minibatches_processed = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        
        # Register handlers
        self._setup_signal_handlers()
        atexit.register(self._at_exit_handler)
        
    def _setup_signal_handlers(self):
        """Setup handlers for all interruption scenarios."""
        
        def sigint_handler(signum, frame):
            """Ctrl+C: once = graceful, twice = force quit"""
            self.ctrl_c_count += 1
            
            if self.ctrl_c_count == 1:
                print(f"\n\n⚠ Ctrl+C pressed once. Will save after current batch.")
                print("Press Ctrl+C again to force quit WITHOUT saving.")
                self.stop_requested = True
            else:
                print(f"\n\n⚠ Ctrl+C pressed twice. Force quitting NOW (no save).")
                sys.exit(1)
        
        def sigterm_handler(signum, frame):
            """SIGTERM: Cloud preemption - save immediately"""
            print(f"\n\n⚠ CLOUD PREEMPTION / SIGTERM received!")
            print("Saving emergency checkpoint NOW...")
            self._emergency_save()
            print("Exiting.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)
    
    def _at_exit_handler(self):
        """Called on Python exit - last resort for power failure."""
        if self.is_running and self.total_minibatches_processed > 0:
            try:
                print(f"\n⚠ ATEXIT: Attempting emergency save...")
                self._emergency_save()
                print("Emergency checkpoint saved.")
            except Exception as e:
                print(f"Could not save on exit: {e}")
    
    def _emergency_save(self):
        """Save checkpoint immediately for emergencies."""
        try:
            dataloader_state = self.dataloader.get_state()
            
            self.state_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                dataloader_state=dataloader_state,
                epoch=self.current_epoch,
                batch_idx=self.current_batch,
                total_minibatches_processed=self.total_minibatches_processed,
                loss=self.current_loss,
                is_emergency=True
            )
        except Exception as e:
            print(f"Emergency save failed: {e}")
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.current_epoch,
                    'total_minibatches_processed': self.total_minibatches_processed,
                }, self.state_manager.save_dir / 'last_resort.pt')
                print("Bare minimum checkpoint saved.")
            except:
                print("CRITICAL: Could not save anything!")
    
    def train(self, num_epochs, resume=True):
        """Train with full interruption resilience."""
        
        # Try to resume from checkpoint
        if resume:
            loaded = self.state_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.dataloader
            )
            
            if loaded:
                (self.start_epoch, self.start_batch, 
                 self.total_minibatches_processed, _, _, is_emergency) = loaded
                
                # Calculate total batches in the saved epoch
                total_batches_in_epoch = self.dataloader.get_total_batches_in_epoch()
                
                # Check if training is actually complete
                training_complete = False
                
                # Case 1: All epochs finished (start_epoch is 0-indexed)
                if self.start_epoch >= num_epochs:
                    training_complete = True
                
                # Case 2: Last epoch had all batches processed
                elif self.start_epoch == num_epochs - 1 and self.start_batch >= total_batches_in_epoch:
                    training_complete = True
                
                if training_complete:
                    print("\n" + "="*60)
                    print("✓ TRAINING ALREADY COMPLETE")
                    print("="*60)
                    print(f"All {num_epochs} epochs finished in previous run.")
                    print(f"Last checkpoint: minibatch {self.total_minibatches_processed}")
                    print(f"Epoch {self.start_epoch+1}/{num_epochs}")
                    print(f"Loss: {self.current_loss:.4f}")
                    print("="*60)
                    print("Nothing to train.")
                    print("Use --clean for fresh start, or increase --epochs.")
                    print("="*60)
                    return
                
                if is_emergency:
                    print(f"\n⚠ Resuming from emergency save.")
                    # Emergency save happens AFTER batch completion (SIGTERM, Ctrl+C)
                    # So the saved batch was already fully processed
                    self.start_batch += 1
                    print(f"  Batch {self.start_batch-1} was completed before save.")
                    print(f"  Starting from batch {self.start_batch}.")
            else:
                print("\nStarting fresh training.")
        
        self.is_running = True
        self.ctrl_c_count = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs} (starting from {self.start_epoch+1})")
        print(f"Total minibatches processed so far: {self.total_minibatches_processed}")
        print(f"Save checkpoint every: {self.state_manager.save_every_n_minibatches} minibatches")
        print(f"Checkpoint directory: {self.state_manager.save_dir}/")
        print(f"{'='*60}")
        print(f"Interruption handling:")
        print(f"  Ctrl+C once  → Graceful stop (save & exit)")
        print(f"  Ctrl+C twice → Force quit (no save)")
        print(f"  Cloud preempt → Emergency save & exit")
        print(f"  Power failure → Emergency save on exit")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(self.start_epoch, num_epochs):
                if not self.is_running:
                    break
                    
                self.current_epoch = epoch
                avg_loss = self.train_epoch(epoch)
                
                if not self.is_running:
                    break
                    
                print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')
                
        except KeyboardInterrupt:
            self._emergency_save()
            
        except Exception as e:
            print(f"\n⚠ Unexpected error: {e}")
            self._emergency_save()
            raise
        
        finally:
            self.is_running = False
            
            if self.stop_requested:
                print(f"\n⏸ Training stopped at minibatch {self.total_minibatches_processed}")
                print(f"   Epoch {self.current_epoch+1}, batch {self.current_batch}")
                print(f"   Resume with: python train.py")
            elif self.current_epoch + 1 >= num_epochs:
                print(f"\n✓ Training complete!")
                print(f"   Total minibatches: {self.total_minibatches_processed}")
                print(f"   All {num_epochs} epochs finished.")
            else:
                print(f"\n⚠ Training ended unexpectedly at minibatch {self.total_minibatches_processed}")
                print(f"   Epoch {self.current_epoch+1}/{num_epochs}")
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        
        # Resume from checkpoint if needed
        if epoch == self.start_epoch and self.start_batch > 0:
            resume_indices = self.dataloader.current_epoch_indices
            resume_batch_start = self.start_batch
        else:
            resume_indices = None
            resume_batch_start = 0
            self.start_batch = 0
        
        self.dataloader.start_epoch(
            epoch, 
            resume_indices=resume_indices,
            resume_batch_start=resume_batch_start
        )
        
        total_batches = self.dataloader.get_total_batches_in_epoch()
        total_samples = len(self.dataloader.current_epoch_indices)
        
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch+1}',
                    initial=resume_batch_start)
        
        batch_idx = resume_batch_start
        total_loss = 0.0
        batches_in_epoch = 0
        
        while True:
            # Check for stop request (Ctrl+C once)
            if self.stop_requested:
                self._emergency_save()
                pbar.close()
                self.is_running = False
                return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0
            
            # Get batch
            batch, wait_time, load_time = self.dataloader.get_next_batch()
            
            if batch is None:
                break
            
            # Compute
            compute_start = time.time()
            
            X, Y = batch
            X, Y = X.to(self.device), Y.to(self.device)
            
            # Check if X is 3D (batch, features, time) and transpose if needed
            if X.dim() == 3:
                # Model expects (B, T, F) but we have (B, F, T)
                # Transpose to (B, T, F)
                X = X.transpose(1, 2)  # (B, F, T) -> (B, T, F)
            # If X is 2D (batch, features), keep as is for dummy model
            
            self.optimizer.zero_grad()
            P = self.model(X)
            L = self.loss_fn(P, X, Y)
            L.backward()
            self.optimizer.step()
            
            compute_time = time.time() - compute_start
            
            # Update counters
            batch_idx += 1
            batches_in_epoch += 1
            self.total_minibatches_processed += 1
            total_loss += L.item()
            avg_loss = total_loss / batches_in_epoch
            self.current_batch = batch_idx
            self.current_loss = avg_loss
            
            # Progress bar - cap sample count at total
            samples_used = min(batch_idx * self.dataloader.batch_size, total_samples)
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'wait': f'{wait_time*1000:.0f}ms',
                'comp': f'{compute_time*1000:.0f}ms',
                'data': f'{samples_used}/{total_samples}',
                'total': self.total_minibatches_processed
            })
            
            # Periodic checkpoint
            if (self.total_minibatches_processed % 
                self.state_manager.save_every_n_minibatches == 0):
                
                dataloader_state = self.dataloader.get_state()
                
                self.state_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=dataloader_state,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_minibatches_processed=self.total_minibatches_processed,
                    loss=avg_loss,
                    is_emergency=False
                )
        
        # Ensure progress bar shows completed
        pbar.update(total_batches - pbar.n)
        pbar.close()
        
        return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINER MODULE DEMO")
    print("=" * 60)
    
    # Simple test setup with 3D data to match the real use case
    class DummyModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            # For 3D input (B, T, F), we flatten and project
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            # If x is 3D (B, T, F), flatten to (B, T*F)
            if x.dim() == 3:
                x = x.reshape(x.size(0), -1)  # Use reshape instead of view
            return self.fc(x)
    
    # Create dummy data
    input_dim = 10  # This is T * F for 3D data
    output_dim = 2
    batch_size = 4
    T = 5   # time steps
    F = 2   # features
    total_samples = 100
    
    model = DummyModel(T * F, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy dataset that returns 3D tensors (B, F, T)
    class DummyDataset:
        def __len__(self): 
            return total_samples
        def __getitem__(self, idx): 
            # Return (B, F, T) shape to simulate real data
            x = torch.randn(F, T)  # (F, T)
            y = torch.randn(output_dim)
            return x, y
    
    class DummyLoader:
        def __init__(self):
            self.dataset = DummyDataset()
            self.batch_size = batch_size
            self.current_epoch_indices = list(range(total_samples))
            self.current_batch_start = 0
            self.dataset_len = total_samples
        
        def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
            self.current_batch_start = resume_batch_start
            if resume_indices is not None:
                self.current_epoch_indices = resume_indices
        
        def get_total_batches_in_epoch(self):
            return math.ceil(self.dataset_len / self.batch_size)
        
        def get_next_batch(self):
            if self.current_batch_start >= self.get_total_batches_in_epoch():
                return None, 0, 0
            
            # Generate batch of 3D data (B, F, T)
            remaining = self.dataset_len - self.current_batch_start * self.batch_size
            batch_size = min(self.batch_size, remaining)
            X_batch = torch.randn(batch_size, F, T)
            Y_batch = torch.randn(batch_size, output_dim)
            self.current_batch_start += 1
            return (X_batch, Y_batch), 0.001, 0.001
        
        def get_state(self):
            return {
                'epoch_indices': self.current_epoch_indices,
                'batch_start': self.current_batch_start
            }
        
        def restore_state(self, state):
            self.current_epoch_indices = state['epoch_indices']
            self.current_batch_start = state['batch_start']
    
    dataloader = DummyLoader()
    
    def dummy_loss(P, X, Y):
        return nn.functional.mse_loss(P, Y)
    
    # Create trainer
    trainer = ResumableTrainer(
        model=model,
        loss_fn=dummy_loss,
        optimizer=optimizer,
        dataloader=dataloader,
        save_dir='test_checkpoints',
        device='cpu',
        save_every_n_minibatches=5
    )
    
    print("\nRunning test training for 2 epochs...")
    trainer.train(num_epochs=2, resume=False)
    
    # Clean up test checkpoints
    import shutil
    if Path('test_checkpoints').exists():
        shutil.rmtree('test_checkpoints')
        print("\nCleaned up test checkpoints.")
    
    print("\nTrainer module demo complete!")