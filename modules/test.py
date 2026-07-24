import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import queue
from tqdm import tqdm
import time
import os
import json
from pathlib import Path
import random
import signal
import sys
import atexit
import math
import numpy as np
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# TRADING DATASET (Replaces SimpleDataset)
# ═══════════════════════════════════════════════════════════

class TradingDataset(Dataset):
    """Trading dataset that implements torch.utils.data.Dataset interface."""
    
    def __init__(
        self, 
        features: List[np.ndarray], 
        price_feature_idx: int,
        past_days: int = 30,
        future_days: int = 5,
        step: int = 1
    ):
        """
        Initialize the trading dataset.
        
        Args:
            features: List of F numpy float arrays, each of shape (T,)
            price_feature_idx: Index of the price feature in the features list
            past_days: Number of days of past data (each day = 60*24 minutes)
            future_days: Number of days of future data for prediction
            step: Step size for start indices (default: 1 = every minute)
        """
        self.past_days = past_days
        self.future_days = future_days
        self.price_feature_idx = price_feature_idx
        self.step = step
        
        # Stack features into (F, T) tensor
        self.Total = self._stack_features(features)
        self.F, self.T = self.Total.shape
        
        # Calculate window sizes in minutes
        self.past_window = 60 * 24 * past_days  # minutes in past_days
        self.future_window = 60 * 24 * future_days  # minutes in future_days
        self.total_window = self.past_window + self.future_window
        
        # Calculate valid start indices with step
        self.valid_starts = self._get_valid_starts()
        
        print(f"TradingDataset initialized:")
        print(f"  Features: {self.F}, Time steps: {self.T}")
        print(f"  Past window: {self.past_window} minutes ({past_days} days)")
        print(f"  Future window: {self.future_window} minutes ({future_days} days)")
        print(f"  Step: {step}")
        print(f"  Total samples: {len(self)}")
        
    def _stack_features(self, features: List[np.ndarray]) -> torch.Tensor:
        """Stack numpy arrays into a torch tensor."""
        shapes = [arr.shape for arr in features]
        if len(set(shapes)) != 1:
            raise ValueError(f"All features must have the same shape. Got: {shapes}")
        
        stacked = np.stack(features, axis=0)
        return torch.from_numpy(stacked).float()
    
    def _get_valid_starts(self) -> List[int]:
        """Get all valid start indices with step."""
        max_start = self.T - self.total_window
        if max_start < 0:
            raise ValueError(
                f"Time series length {self.T} is too short for "
                f"past {self.past_window} + future {self.future_window} = {self.total_window} minutes"
            )
        
        # Get starts from 0 to max_start with the specified step
        starts = list(range(0, max_start + 1, self.step))
        
        if not starts:
            raise ValueError(f"No valid starts with step {self.step}")
        
        return starts
    
    def __len__(self) -> int:
        """Number of valid samples."""
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample (x, y) for a given index.
        
        Args:
            idx: Index into valid_starts
            
        Returns:
            x: Tensor of shape (F, past_window)
            y: Tensor of shape (future_window,) - price feature only
        """
        start = self.valid_starts[idx]
        
        # Extract x (past data)
        x = self.Total[:, start:start + self.past_window]
        
        # Extract y (future price data)
        y_start = start + self.past_window
        y_end = y_start + self.future_window
        y = self.Total[self.price_feature_idx, y_start:y_end]
        
        return x, y


# ═══════════════════════════════════════════════════════════
# THREADED DATA LOADER
# ═══════════════════════════════════════════════════════════

class ThreadedDataLoader:
    """DataLoader that pre-fetches next batch in a separate thread."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current_epoch_indices = None
        self.current_batch_start = 0
        
        self.iterator = None
        self.prefetch_queue = queue.Queue(maxsize=1)
        self.prefetch_thread = None
        self.load_start_time = None
        
    def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
        self.epoch = epoch
        
        if resume_indices is not None:
            self.current_epoch_indices = resume_indices
        else:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(self.current_epoch_indices)
        
        self.current_batch_start = resume_batch_start
        self._create_iterator()
        self._start_prefetch()
    
    def _create_iterator(self):
        start = self.current_batch_start * self.batch_size
        remaining_indices = self.current_epoch_indices[start:]
        self.iterator = self._batch_generator(remaining_indices)
    
    def _batch_generator(self, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            X_list = []
            Y_list = []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                X_list.append(x)
                Y_list.append(y)
            
            X_batch = torch.stack(X_list)
            Y_batch = torch.stack(Y_list)
            
            yield X_batch, Y_batch
    
    def get_total_batches_in_epoch(self):
        """Return total number of batches in current epoch (including partial last batch)."""
        if self.current_epoch_indices is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        return math.ceil(len(self.current_epoch_indices) / self.batch_size)
    
    def get_state(self):
        return {
            'epoch_indices': self.current_epoch_indices,
            'batch_start': self.current_batch_start,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
        }
    
    def restore_state(self, state):
        self.current_epoch_indices = state['epoch_indices']
        self.current_batch_start = state['batch_start']
    
    def _start_prefetch(self):
        self.load_start_time = time.time()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
    def _prefetch_worker(self):
        try:
            batch = next(self.iterator)
            load_time = time.time() - self.load_start_time
            self.prefetch_queue.put(('data', batch, load_time))
        except StopIteration:
            self.prefetch_queue.put(('stop', None, 0))
        except Exception as e:
            self.prefetch_queue.put(('error', e, 0))
    
    def get_next_batch(self):
        wait_start = time.time()
        status, data, load_time = self.prefetch_queue.get()
        wait_time = time.time() - wait_start
        
        if status == 'stop':
            return None, wait_time, load_time
        elif status == 'error':
            raise data
        
        self.current_batch_start += 1
        self._start_prefetch()
        return data, wait_time, load_time


# ═══════════════════════════════════════════════════════════
# TRAINING STATE MANAGER (FIXED PERMISSION ISSUE)
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
    Trainer with robust handling of all interruption scenarios:
    - Normal run: Continuous training with periodic checkpoints
    - Ctrl+C once: Graceful stop after current batch (save & exit)
    - Ctrl+C twice: Force quit without saving
    - SIGTERM: Cloud preemption - emergency save immediately
    - Power failure: atexit emergency save (best effort)
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
# MODEL (Modified for TradingDataset)
# ═══════════════════════════════════════════════════════════

class SimpleModel(nn.Module):
    """Simple neural network model adapted for trading data."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, F, past_window)
        # Flatten: (batch, F * past_window)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def custom_loss(P, X, Y):
    """Custom loss function for trading predictions."""
    # P: (batch, future_window)
    # Y: (batch, future_window)
    return nn.functional.mse_loss(P, Y)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resilient Trading Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save every N minibatches')
    parser.add_argument('--clean', action='store_true', help='Delete old checkpoints')
    parser.add_argument('--past-days', type=int, default=30, help='Past days for input')
    parser.add_argument('--future-days', type=int, default=5, help='Future days for prediction')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--num-features', type=int, default=10, help='Number of features')
    parser.add_argument('--total-days', type=int, default=100, help='Total days of data')
    parser.add_argument('--step', type=int, default=10, help='Step size for start indices (e.g., 60 = every hour)')
    args = parser.parse_args()
    
    CHECKPOINT_DIR = 'trading_checkpoints'
    
    # Clean start if requested
    if args.clean and Path(CHECKPOINT_DIR).exists():
        import shutil
        shutil.rmtree(CHECKPOINT_DIR)
        print("Cleaned previous checkpoints.\n")
    
    # Create synthetic trading data
    T = 60 * 24 * args.total_days  # total_days of minute data
    F = args.num_features
    
    print(f"Generating synthetic trading data: {T} minutes, {F} features")
    
    # Create features
    features = []
    for i in range(F):
        if i == 0:  # Price feature - random walk
            price = np.cumsum(np.random.randn(T) * 0.01)
            features.append(price.astype(np.float32))
        elif i == 1:  # Volume - positive
            volume = np.abs(np.random.randn(T) * 100 + 1000)
            features.append(volume.astype(np.float32))
        else:  # Other features
            features.append(np.random.randn(T).astype(np.float32))
    
    # Create dataset with step
    dataset = TradingDataset(
        features=features,
        price_feature_idx=0,
        past_days=args.past_days,
        future_days=args.future_days,
        step=args.step
    )
    
    # Calculate dimensions
    past_window = 60 * 24 * args.past_days
    future_window = 60 * 24 * args.future_days
    F = len(features)
    
    print(f"\nData shapes:")
    print(f"  Input x: (F={F}, past_window={past_window})")
    print(f"  Output y: (future_window={future_window})")
    print(f"  Total samples: {len(dataset)}")
    
    # Create model - input is flattened F * past_window
    model = SimpleModel(
        input_dim=F * past_window,
        hidden_dim=args.hidden_dim,
        output_dim=future_window
    )
    
    # Create dataloader
    dataloader = ThreadedDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = ResumableTrainer(
        model=model,
        loss_fn=custom_loss,
        optimizer=optimizer,
        dataloader=dataloader,
        save_dir=CHECKPOINT_DIR,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_every_n_minibatches=args.save_every
    )
    
    # Show status
    print("\n" + "=" * 60)
    print("TRADING MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} samples, batch size: {args.batch_size}")
    print(f"Past days: {args.past_days} ({past_window} minutes)")
    print(f"Future days: {args.future_days} ({future_window} minutes)")
    print(f"Step: {args.step} minute(s)")
    print(f"Epochs: {args.epochs}")
    print(f"Minibatches per epoch: {math.ceil(len(dataset) / args.batch_size)}")
    print(f"Model: Simple MLP with {args.hidden_dim} hidden units")
    print(f"Input dimension: {F * past_window} (flattened F×past_window)")
    print(f"Output dimension: {future_window}")
    print(f"Device: {trainer.device}")
    print(f"Save every: {args.save_every} minibatches")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print("=" * 60)
    
    # Check for existing checkpoint
    state_file = Path(CHECKPOINT_DIR) / 'training_state.json'
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"Found previous checkpoint at minibatch {state['total_minibatches_processed']}")
            print(f"Epoch {state['epoch']+1}, batch {state['batch_idx']}")
        except:
            pass
    else:
        print("No previous checkpoint - fresh start")
    
    print()
    
    # Train
    trainer.train(num_epochs=args.epochs, resume=True)