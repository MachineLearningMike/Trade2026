# trainer.py
# Resumable trainer with training_finished flag in checkpoint state and validation

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
import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from config import get_config


# ═══════════════════════════════════════════════════════════
# CHECKPOINT KEY GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_checkpoint_key(config: dict, dataset_signature: str) -> str:
    """
    Hash everything that makes model weights / optimizer state non-transferable.

    `dataset_signature` already covers data source, windows, labels, splits,
    and normalization — do not re-list those here.
    """
    key_params = {
        # Data + labels (from TradingDataset._generate_cache_key)
        'dataset_signature': dataset_signature,

        # Architecture (changes weight shapes / forward pass)
        'patch_size': config.get('patch_size'),
        'num_layers': config.get('num_layers'),
        'num_heads': config.get('num_heads'),
        'n_latent_macro_features': config.get('n_latent_macro_features'),
        'ff_dim_multiplier': config.get('ff_dim_multiplier'),
        'dropout': config.get('dropout'),
        'head_hidden_dim_multiplier': config.get('head_hidden_dim_multiplier'),
        'output_dim': config.get('output_dim', 3),
        'causal': config.get('causal'),
        'use_reconstruction': config.get('use_reconstruction'),

        # Objective + optimizer (same weights, different training meaning)
        'slope_loss_weight': config.get('slope_loss_weight'),
        'intercept_loss_weight': config.get('intercept_loss_weight'),
        'reconstruction_loss_weight': config.get('reconstruction_loss_weight'),
        'learning_rate': config.get('learning_rate'),
        'batch_size': config.get('batch_size'),
    }

    param_string = json.dumps(key_params, sort_keys=True, default=str)
    return hashlib.md5(param_string.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════
# TRAINING STATE MANAGER
# ═══════════════════════════════════════════════════════════

class TrainingState:
    """Manages saving and loading of complete training state with atomic writes."""
    
    def __init__(self, save_dir='trading_checkpoints', save_every_n_minibatches=100, 
                 max_checkpoints=5, checkpoint_key=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_minibatches = save_every_n_minibatches
        self.max_checkpoints = max_checkpoints  # ← This was missing
        
        if checkpoint_key:
            self.save_dir = self.save_dir / checkpoint_key
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.save_dir / 'training_state.json'
        self.checkpoint_key = checkpoint_key
        self.planned_total_epochs = None  # preserved across checkpoint saves
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoint files, keeping only the most recent ones."""
        all_checkpoints = sorted(
            list(self.save_dir.glob('checkpoint_minibatch*.pt')) + 
            list(self.save_dir.glob('emergency_minibatch*.pt')),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Keep only the most recent max_checkpoints files
        for old_file in all_checkpoints[self.max_checkpoints:]:
            try:
                old_file.unlink()
                print(f"🧹 Removed old checkpoint: {old_file.name}")
            except Exception as e:
                print(f"⚠ Failed to remove old checkpoint {old_file.name}: {e}")
        
    def save_checkpoint(self, model, optimizer, dataloader_state, epoch, batch_idx, 
                        total_minibatches_processed, loss, is_emergency=False, training_finished=False,
                        val_loss=None, history=None, embed_dim=None, max_seq_len=None,
                        total_epochs=None):
        """Save checkpoint with atomic write to prevent corruption."""
        # Preserve planned total_epochs across saves (was previously wiped every checkpoint).
        preserved_total = total_epochs
        if preserved_total is None and self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    prev = json.load(f)
                preserved_total = prev.get('total_epochs')
            except Exception:
                preserved_total = None
        if preserved_total is None:
            preserved_total = getattr(self, 'planned_total_epochs', None)

        checkpoint = {
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'val_loss': val_loss,
            'history': history,
            'training_finished': training_finished,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataloader_state': dataloader_state,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
            'checkpoint_key': self.checkpoint_key,
            'embed_dim': embed_dim,
            'max_seq_len': max_seq_len,
            'total_epochs': preserved_total,
        }
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        prefix = 'emergency' if is_emergency else 'checkpoint'
        checkpoint_file = self.save_dir / f'{prefix}_minibatch{total_minibatches_processed}_{timestamp}.pt'
        
        temp_file = checkpoint_file.with_suffix('.tmp')
        torch.save(checkpoint, temp_file)
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(temp_file, checkpoint_file)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    try:
                        import shutil
                        shutil.copy2(temp_file, checkpoint_file)
                        os.remove(temp_file)
                    except:
                        raise e
        
        state_info = {
            'latest_checkpoint': str(checkpoint_file),
            'version': 4,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches_processed': total_minibatches_processed,
            'loss': loss,
            'val_loss': val_loss,
            'history': history,
            'training_finished': training_finished,
            'timestamp': time.time(),
            'is_emergency': is_emergency,
            'checkpoint_key': self.checkpoint_key,
            'total_epochs': preserved_total,
        }
        if training_finished:
            # epoch is 0-based index of last trained epoch
            state_info['completed_epochs'] = int(epoch) + 1
        
        temp_state = self.state_file.with_suffix('.tmp')
        with open(temp_state, 'w') as f:
            json.dump(state_info, f, indent=2, default=str)
        
        for attempt in range(max_retries):
            try:
                os.replace(temp_state, self.state_file)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    fallback_file = self.save_dir / f'training_state_{timestamp}.json'
                    with open(fallback_file, 'w') as f:
                        json.dump(state_info, f, indent=2, default=str)
                    print(f"⚠ State file saved as fallback: {fallback_file}")
                    break
        
        self._cleanup_old_checkpoints()
        
        if training_finished:
            print(f"\n{'='*60}")
            print(f"✓ TRAINING COMPLETE - Final checkpoint saved!")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Total minibatches: {total_minibatches_processed}")
            print(f"  Loss: {loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"{'='*60}")
        elif is_emergency:
            print(f"\n{'!'*60}")
            print(f"EMERGENCY CHECKPOINT SAVED at minibatch {total_minibatches_processed}")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            print(f"{'!'*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Checkpoint saved at minibatch {total_minibatches_processed}")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  File: {checkpoint_file.name}")
            print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"{'='*60}")
    
    def mark_training_finished(self, model, optimizer, dataloader_state, epoch, batch_idx, 
                               total_minibatches_processed, loss, val_loss=None, history=None,
                               embed_dim=None, max_seq_len=None):
        """Save a final checkpoint marking training as finished."""
        self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            dataloader_state=dataloader_state,
            epoch=epoch,
            batch_idx=batch_idx,
            total_minibatches_processed=total_minibatches_processed,
            loss=loss,
            val_loss=val_loss,
            history=history,
            is_emergency=False,
            training_finished=True,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
    
    def is_training_finished(self):
        """Check if training has been marked as finished by reading the state file."""
        if not self.state_file.exists():
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
            return state_info.get('training_finished', False)
        except:
            return False
    
    def clear_finished_flag(self):
        """Clear the training_finished flag (used for --extend)."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
            
            state_info['training_finished'] = False
            
            temp_state = self.state_file.with_suffix('.tmp')
            with open(temp_state, 'w') as f:
                json.dump(state_info, f, indent=2, default=str)
            
            for attempt in range(5):
                try:
                    os.replace(temp_state, self.state_file)
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        time.sleep(0.1 * (2 ** attempt))
                    else:
                        raise
            
            print(f"✓ Cleared training_finished flag for extension (Key: {self.checkpoint_key}).")
        except Exception as e:
            print(f"⚠ Failed to clear finished flag: {e}")
    
    def get_progress_epochs(
        self,
        total_batches_in_epoch: Optional[int] = None,
        continuous: bool = False,
    ) -> float:
        """
        Training progress from training_state.json.

        Checkpoint fields:
          epoch     — 0-based index of the epoch being trained when saved
          batch_idx — number of batches already finished in that epoch

        continuous=False: fully completed whole epochs.
        continuous=True:  epoch + batch_idx/total_batches (for --extend clamp).
        """
        if not self.state_file.exists():
            return 0.0

        try:
            with open(self.state_file, "r") as f:
                state_info = json.load(f)
        except Exception:
            return 0.0

        if state_info.get("training_finished", False):
            return float(state_info.get(
                "completed_epochs",
                int(state_info.get("epoch", 0)) + 1,
            ))

        epoch = int(state_info.get("epoch", 0))
        batch_idx = int(state_info.get("batch_idx", 0))
        if total_batches_in_epoch is not None and total_batches_in_epoch > 0:
            if batch_idx >= total_batches_in_epoch:
                return float(epoch + 1)
            if continuous:
                return float(epoch) + float(batch_idx) / float(total_batches_in_epoch)
            return float(epoch)
        return float(epoch)

    def get_completed_epochs(self):
        """Backward-compatible alias for get_progress_epochs(). """
        return self.get_progress_epochs()
    
    def get_total_epochs(self) -> float:
        """Get the planned total epochs from the state file (0 if unknown)."""
        if not self.state_file.exists():
            return 0.0
        
        try:
            with open(self.state_file, "r") as f:
                state_info = json.load(f)
            te = state_info.get("total_epochs")
            return float(te) if te is not None else 0.0
        except Exception:
            return 0.0
    
    def update_total_epochs(self, total_epochs):
        """Update the total epochs in the state file (create file if missing)."""
        total_epochs = float(total_epochs)
        self.planned_total_epochs = total_epochs

        state_info = {}
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state_info = json.load(f)
            except Exception:
                state_info = {}

        state_info["total_epochs"] = total_epochs
        state_info["checkpoint_key"] = self.checkpoint_key

        try:
            temp_state = self.state_file.with_suffix(".tmp")
            with open(temp_state, "w") as f:
                json.dump(state_info, f, indent=2, default=str)

            for attempt in range(5):
                try:
                    os.replace(temp_state, self.state_file)
                    break
                except (PermissionError, OSError):
                    if attempt < 4:
                        time.sleep(0.1 * (2 ** attempt))
                    else:
                        raise

            print(f"✓ Updated total_epochs to {total_epochs:g} (Key: {self.checkpoint_key}).")
        except Exception as e:
            print(f"⚠ Failed to update total_epochs: {e}")
    
    def load_latest_checkpoint(self, model, optimizer, dataloader):
        """Load the latest checkpoint from the most recent file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("⚠ State file corrupted. Starting fresh.")
            return None
        
        # Find the most recent checkpoint file FIRST (we need it regardless)
        checkpoint_files = list(self.save_dir.glob('checkpoint_minibatch*.pt'))
        emergency_files = list(self.save_dir.glob('emergency_minibatch*.pt'))
        all_files = checkpoint_files + emergency_files
        
        if not all_files:
            print(f"⚠ No checkpoint files found in {self.save_dir}. Starting fresh.")
            return None
        
        latest_checkpoint = max(all_files, key=lambda f: f.stat().st_mtime)
        print(f"Found checkpoint: {latest_checkpoint.name}")
        
        # Always load the checkpoint to get the full data
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}. Starting fresh.")
            return None
        
        # Extract data from checkpoint
        epoch = checkpoint.get('epoch', 0)
        batch_idx = checkpoint.get('batch_idx', 0)
        total_minibatches = checkpoint.get('total_minibatches_processed', 0)
        loss = checkpoint.get('loss', 0.0)
        val_loss = checkpoint.get('val_loss', None)
        history = checkpoint.get('history', None)
        is_emergency = checkpoint.get('is_emergency', False)
        dataloader_state = checkpoint.get('dataloader_state', None)
        
        # Check if training is finished
        if state_info.get('training_finished', False):
            print(f"✓ Training is already finished!")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  Final checkpoint: {latest_checkpoint.name}")
            print(f"  Total minibatches: {total_minibatches}")
            print(f"  Loss: {loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            
            # Even though training is finished, we return the checkpoint data
            # So the caller knows what was completed
            return {
                'status': 'finished',
                'epoch': epoch,
                'batch_idx': batch_idx,
                'total_minibatches': total_minibatches,
                'loss': loss,
                'val_loss': val_loss,
                'history': history,
                'dataloader_state': dataloader_state,
                'is_emergency': is_emergency,
            }
        
        # Training is NOT finished - load model and optimizer
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✓ Loaded checkpoint with strict matching.")
        except RuntimeError as e:
            print(f"⚠ Strict loading failed. Attempting partial load...")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False
                )
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)} (will be initialized randomly)")
                    if len(missing_keys) <= 5:
                        print(f"    Missing: {missing_keys}")
                    else:
                        print(f"    First 5 missing keys: {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"  Unexpected keys: {len(unexpected_keys)} (ignored)")
                    if len(unexpected_keys) <= 5:
                        print(f"    Unexpected: {unexpected_keys}")
                    else:
                        print(f"    First 5 unexpected keys: {unexpected_keys[:5]}")
                print("✓ Loaded checkpoint with partial matching.")
            except Exception as e2:
                print(f"⚠ Failed to load model state: {e2}. Starting fresh.")
                return None
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Loaded optimizer state.")
        except Exception as e:
            print(f"⚠ Failed to load optimizer state: {e}. Starting fresh.")
            return None
        
        if dataloader_state is not None:
            dataloader.restore_state(dataloader_state)
        
        if is_emergency:
            print(f"⚠ Resuming from EMERGENCY checkpoint")
        else:
            print(f"Resuming from checkpoint")
        print(f"  Key: {self.checkpoint_key}")
        print(f"  Minibatch: {total_minibatches}")
        print(f"  Epoch {epoch+1}, batch {batch_idx}")
        print(f"  Loss: {loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        
        return {
            'status': 'checkpoint',
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_minibatches': total_minibatches,
            'loss': loss,
            'val_loss': val_loss,
            'history': history,
            'dataloader_state': dataloader_state,
            'is_emergency': is_emergency,
        }

# ═══════════════════════════════════════════════════════════
# TRAINING HISTORY
# ═══════════════════════════════════════════════════════════

class TrainingHistory:
    """Track and visualize training history."""
    
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.timestamps = []
    
    def add_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss if val_loss is not None else None)
        self.timestamps.append(time.time())
    
    def to_dict(self) -> Dict[str, List]:
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamps': self.timestamps,
        }
    
    def from_dict(self, data: Dict[str, List]):
        self.epochs = data.get('epochs', [])
        self.train_losses = data.get('train_losses', [])
        self.val_losses = data.get('val_losses', [])
        self.timestamps = data.get('timestamps', [])
    
    def plot(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)):
        if not self.epochs:
            print("No training history to plot.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.epochs, self.train_losses, 'b-', linewidth=2, label='Training Loss')
        
        val_losses = [v for v in self.val_losses if v is not None]
        if val_losses:
            val_epochs = [self.epochs[i] for i, v in enumerate(self.val_losses) if v is not None]
            ax.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        min_train = min(self.train_losses)
        min_train_epoch = self.epochs[self.train_losses.index(min_train)]
        ax.annotate(f'Min Train: {min_train:.4f}', 
                   xy=(min_train_epoch, min_train),
                   xytext=(min_train_epoch, min_train * 1.1),
                   fontsize=9, color='blue')
        
        if val_losses:
            min_val = min(val_losses)
            min_val_epoch = val_epochs[val_losses.index(min_val)]
            ax.annotate(f'Min Val: {min_val:.4f}', 
                       xy=(min_val_epoch, min_val),
                       xytext=(min_val_epoch, min_val * 1.1),
                       fontsize=9, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Training history plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def print_summary(self):
        if not self.epochs:
            print("No training history available.")
            return
        
        print("\n" + "="*60)
        print("TRAINING HISTORY SUMMARY")
        print("="*60)
        print(f"Total epochs: {len(self.epochs)}")
        print(f"Final training loss: {self.train_losses[-1]:.6f}")
        print(f"Best training loss: {min(self.train_losses):.6f} (epoch {self.epochs[self.train_losses.index(min(self.train_losses))]})")
        
        val_losses = [v for v in self.val_losses if v is not None]
        if val_losses:
            val_epochs = [self.epochs[i] for i, v in enumerate(self.val_losses) if v is not None]
            print(f"Final validation loss: {val_losses[-1]:.6f}")
            print(f"Best validation loss: {min(val_losses):.6f} (epoch {val_epochs[val_losses.index(min(val_losses))]})")
        print("="*60)


# ═══════════════════════════════════════════════════════════
# RESUMABLE TRAINER - OPTIMIZED
# ═══════════════════════════════════════════════════════════

class ResumableTrainer:
    """
    Trainer with robust handling of all interruption scenarios and validation.
    Optimized for speed with reduced overhead.
    """
    
    def __init__(self, dataset_signature,
                model, loss_fn, optimizer, dataloader, 
                save_dir='trading_checkpoints', device='cpu',
                save_every_n_minibatches=100,
                val_loader=None, 
                enable_validation=False, 
                validate_every_n_epochs=1,
                checkpoint_key=None,
                embed_dim=None,
                max_seq_len=None):
        self.dataset_signature = dataset_signature
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.enable_validation = enable_validation
        self.validate_every_n_epochs = validate_every_n_epochs
        
        if checkpoint_key is None:
            config = get_config()
            checkpoint_key = generate_checkpoint_key(config, self.dataset_signature)
        
        self.checkpoint_key = checkpoint_key
        
        self.state_manager = TrainingState(
            save_dir=save_dir,
            save_every_n_minibatches=save_every_n_minibatches,
            checkpoint_key=checkpoint_key
        )
        
        self.stop_requested = False
        self.is_running = False
        self.ctrl_c_count = 0
        
        self.total_minibatches_processed = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        self.current_val_loss = None
        
        self.history = TrainingHistory()
        
        self._setup_signal_handlers()
        atexit.register(self._at_exit_handler)
        
        print(f"🔑 Checkpoint Key: {checkpoint_key}")
        print(f"📁 Checkpoint Directory: {self.state_manager.save_dir}")
        
    def _setup_signal_handlers(self):
        def sigint_handler(signum, frame):
            self.ctrl_c_count += 1
            if self.ctrl_c_count == 1:
                print(f"\n\n⚠ Ctrl+C pressed once. Will save after current batch.")
                print("Press Ctrl+C again to force quit WITHOUT saving.")
                self.stop_requested = True
            else:
                print(f"\n\n⚠ Ctrl+C pressed twice. Force quitting NOW (no save).")
                sys.exit(1)
        
        def sigterm_handler(signum, frame):
            print(f"\n\n⚠ CLOUD PREEMPTION / SIGTERM received!")
            print("Saving emergency checkpoint NOW...")
            self._emergency_save()
            print("Exiting.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)
    
    def _at_exit_handler(self):
        if self.is_running and self.total_minibatches_processed > 0:
            try:
                print(f"\n⚠ ATEXIT: Attempting emergency save...")
                self._emergency_save()
                print("Emergency checkpoint saved.")
            except Exception as e:
                print(f"Could not save on exit: {e}")
    
    def _emergency_save(self):
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
                val_loss=self.current_val_loss,
                history=self.history.to_dict(),
                is_emergency=True,
                training_finished=False,
                embed_dim=getattr(self, 'embed_dim', None),
                max_seq_len=getattr(self, 'max_seq_len', None),
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
    
    def run_validation(self, loader=None, desc="Validation"):
        if loader is None:
            loader = self.val_loader
        
        if loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        loader.start_epoch(self.current_epoch)
        total_batches = loader.get_total_batches_in_epoch()
        
        with torch.no_grad():
            # Create progress bar with proper settings
            pbar = tqdm(total=total_batches, desc=desc, 
                        mininterval=0.5, miniters=1,  # Update more frequently
                        smoothing=0.1)  # Smooth the rate calculation
            
            while True:
                batch, load_time = loader.get_next_batch()
                if batch is None:
                    break
                
                x, y, mean, std, shape = batch
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mean = mean.to(self.device, non_blocking=True)
                std = std.to(self.device, non_blocking=True)
                shape = shape.to(self.device, non_blocking=True)
                
                P = self.model(x, mean, std, shape)
                L = self.loss_fn(P, x, y)
                
                total_loss += L.item()
                num_batches += 1
                
                # Update progress bar every batch
                avg_loss = total_loss / num_batches
                pbar.update(1)  # Always update by 1 batch
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'batch': f'{num_batches}/{total_batches}'
                })
            
            # Ensure progress bar reaches 100%
            if pbar.n < total_batches:
                pbar.update(total_batches - pbar.n)
            pbar.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def plot_history(self, save_path: Optional[str] = None):
        return self.history.plot(save_path=save_path)
    
    def _show_history_if_available(self):
        if self.history.epochs:
            self.history.print_summary()
            plot_path = self.state_manager.save_dir / 'training_history.png'
            self.history.plot(save_path=str(plot_path))
            return True
        return False
    
    @staticmethod
    def _batches_for_epoch(epoch: int, num_epochs: float, total_batches: int) -> int:
        """How many batches to run in 0-based `epoch` given a (possibly fractional) plan."""
        if total_batches <= 0:
            return 0
        if num_epochs >= float(epoch + 1):
            return total_batches
        if num_epochs <= float(epoch):
            return 0
        frac = float(num_epochs) - float(epoch)
        return max(1, min(total_batches, int(math.ceil(frac * total_batches - 1e-12))))

    def handle_extension(self, extend_epochs: float, original_num_epochs: float) -> float:
        """
        Adjust the planned epoch budget by `extend_epochs` (float, may be negative).

        new_total = max(original_plan, stored_plan, progress) + extend_epochs
        then clamp to >= continuous progress (cannot untrain).

        Examples (plan 200):
          at epoch 50, --extend 10   → 210
          at epoch 50, --extend -10  → 190
          at epoch 50, --extend -200 → 50 (clamped to progress)
          --extend 1.5               → fractional final epoch allowed
        """
        extend_epochs = float(extend_epochs)
        original_num_epochs = float(original_num_epochs)

        if abs(extend_epochs) < 1e-15:
            stored = self.state_manager.get_total_epochs()
            return stored if stored > 0 else original_num_epochs

        try:
            total_batches = self.dataloader.get_total_batches_in_epoch()
        except Exception:
            total_batches = None

        progress = float(self.state_manager.get_progress_epochs(total_batches, continuous=False))
        progress_cont = float(self.state_manager.get_progress_epochs(total_batches, continuous=True))

        if progress == 0 and progress_cont == 0:
            checkpoint_files = (
                list(self.state_manager.save_dir.glob("checkpoint_minibatch*.pt"))
                + list(self.state_manager.save_dir.glob("emergency_minibatch*.pt"))
            )
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
                try:
                    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
                    ep = int(checkpoint.get("epoch", 0))
                    bi = int(checkpoint.get("batch_idx", 0))
                    if total_batches and bi >= total_batches:
                        progress = float(ep + 1)
                        progress_cont = float(ep + 1)
                    else:
                        progress = float(ep)
                        if total_batches and total_batches > 0:
                            progress_cont = float(ep) + float(bi) / float(total_batches)
                        else:
                            progress_cont = float(ep)
                    print(f"  (progress from checkpoint {latest.name}: {progress_cont:g})")
                except Exception:
                    pass

        stored_total = float(self.state_manager.get_total_epochs())
        base = max(float(original_num_epochs), stored_total, progress)
        new_total = base + extend_epochs

        clamped = False
        if new_total < progress_cont:
            new_total = progress_cont
            clamped = True

        # Avoid negative / zero empty plans when nothing trained yet
        if new_total < 0:
            new_total = 0.0
            clamped = True

        label = "EXTENDING" if extend_epochs > 0 else "SHORTENING"
        print(f"\n{'='*60}")
        print(f"⚠ {label} TRAINING PLAN BY {extend_epochs:g} EPOCHS")
        print(f"{'='*60}")
        print(f"  Original plan (--epochs): {original_num_epochs:g}")
        print(f"  Stored plan (state):      {stored_total:g}" if stored_total else "  Stored plan (state):      (none)")
        print(f"  Progress (full epochs):   {progress:g}")
        print(f"  Progress (continuous):    {progress_cont:g}")
        print(f"  Base target:              {base:g}")
        print(f"  Delta:                    {extend_epochs:g}")
        print(f"  New total epochs:         {new_total:g}" + ("  (clamped to progress)" if clamped else ""))
        print(f"{'='*60}\n")

        self.state_manager.update_total_epochs(new_total)
        if self.state_manager.is_training_finished() and new_total > progress_cont + 1e-12:
            self.state_manager.clear_finished_flag()
        return new_total
    
    def train(self, num_epochs, resume=True, extend=False, extend_epochs=0):
        """Train the model with robust checkpointing."""
        original_num_epochs = float(num_epochs)
        extend_epochs = float(extend_epochs) if extend_epochs else 0.0
        extend = bool(extend) or abs(extend_epochs) > 1e-15
        eps = 1e-9

        # ============================================================
        # STEP 1: Handle extension / persist planned total
        # ============================================================
        if abs(extend_epochs) > eps:
            num_epochs = float(self.handle_extension(extend_epochs, original_num_epochs))
        else:
            stored_total = float(self.state_manager.get_total_epochs())
            if stored_total > original_num_epochs + eps:
                print(
                    f"⚠ Using stored total_epochs={stored_total:g} "
                    f"(config --epochs={original_num_epochs:g})."
                )
                num_epochs = stored_total
            else:
                num_epochs = original_num_epochs
                self.state_manager.update_total_epochs(num_epochs)

        if extend and abs(extend_epochs) > eps:
            print(f"\n{'='*60}")
            print(f"⚠ PLAN ADJUSTMENT MODE")
            print(f"{'='*60}")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  Delta: {extend_epochs:g}")
            print(f"  Training until epoch {num_epochs:g} (total).")
            print(f"{'='*60}\n")

        self.state_manager.planned_total_epochs = num_epochs
        
        # ============================================================
        # STEP 2: Load checkpoint if resuming
        # ============================================================
        skip_first_validation = False
        start_epoch_for_loop = 0
        
        if resume:
            result = self.state_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.dataloader
            )
            
            if result:
                self.start_epoch = result["epoch"]
                self.start_batch = result["batch_idx"]
                self.total_minibatches_processed = result["total_minibatches"]
                self.current_val_loss = result["val_loss"]
                if result["history"]:
                    self.history.from_dict(result["history"])
                    print(f"✓ Loaded training history ({len(self.history.epochs)} epochs)")

                total_batches_in_epoch = self.dataloader.get_total_batches_in_epoch()

                if result["status"] == "finished":
                    if not (extend and extend_epochs > eps):
                        # finished + negative/zero extend → nothing to do
                        print("\n" + "=" * 60)
                        print("✓ TRAINING ALREADY FINISHED")
                        print("=" * 60)
                        print(f"Key: {self.checkpoint_key}")
                        print(f"All {self.start_epoch + 1} epochs completed in previous run.")
                        print(f"Current plan: {num_epochs:g}")
                        if extend and extend_epochs < 0:
                            print("  (--extend negative cannot untrain finished work; plan clamped.)")
                        print("Use --extend +N to train more epochs.")
                        print("=" * 60)
                        self._show_history_if_available()
                        return

                    print("⚠ Training was finished; positive --extend continues past the last epoch.")
                    self.state_manager.clear_finished_flag()
                    start_epoch_for_loop = self.start_epoch + 1
                    self.start_batch = 0
                    if start_epoch_for_loop >= num_epochs - eps:
                        print(
                            f"\n⚠ Plan {num_epochs:g} does not go past progress "
                            f"({start_epoch_for_loop})."
                        )
                        print("  Use a larger positive --extend value.")
                        return
                    print(
                        f"✓ Extension resume: starting at epoch {start_epoch_for_loop + 1} "
                        f"/ {num_epochs:g}"
                    )

                elif result["status"] == "checkpoint":
                    epoch_complete = (
                        total_batches_in_epoch > 0
                        and self.start_batch >= total_batches_in_epoch
                    )

                    if epoch_complete:
                        start_epoch_for_loop = self.start_epoch + 1
                        self.start_batch = 0
                        skip_first_validation = not (extend and extend_epochs > eps)
                        print(
                            f"⚠ Resuming after completed epoch {self.start_epoch + 1} "
                            f"→ next epoch {start_epoch_for_loop + 1}"
                        )
                    else:
                        start_epoch_for_loop = self.start_epoch
                        skip_first_validation = False
                        print(
                            f"⚠ Resuming mid-epoch {self.start_epoch + 1}, "
                            f"batch {self.start_batch}/{total_batches_in_epoch}"
                        )

                    if result.get("is_emergency"):
                        print(f"\n⚠ Resuming from emergency save at batch {self.start_batch}.")

                    # Continuous progress vs plan
                    if epoch_complete:
                        cont = float(start_epoch_for_loop)
                    else:
                        cont = float(self.start_epoch) + (
                            float(self.start_batch) / float(total_batches_in_epoch)
                            if total_batches_in_epoch else 0.0
                        )

                    if cont >= num_epochs - eps:
                        if extend and extend_epochs > eps:
                            print(
                                f"\n⚠ Progress {cont:g} already at/past plan {num_epochs:g}."
                            )
                            print("  Use a larger positive --extend value.")
                            return
                        # Plan shortened to current progress (or already done)
                        self.state_manager.mark_training_finished(
                            model=self.model,
                            optimizer=self.optimizer,
                            dataloader_state=self.dataloader.get_state(),
                            epoch=self.start_epoch,
                            batch_idx=self.start_batch,
                            total_minibatches_processed=self.total_minibatches_processed,
                            loss=self.current_loss if self.current_loss != 0 else 0,
                            val_loss=self.current_val_loss,
                            history=self.history.to_dict(),
                            embed_dim=getattr(self, "embed_dim", None),
                            max_seq_len=getattr(self, "max_seq_len", None),
                        )
                        print("\n" + "=" * 60)
                        print("✓ TRAINING COMPLETE FOR CURRENT PLAN")
                        print("=" * 60)
                        print(f"Key: {self.checkpoint_key}")
                        print(f"Progress {cont:g} >= plan {num_epochs:g}.")
                        print("Use --extend +N to train further.")
                        print("=" * 60)
                        self._show_history_if_available()
                        return
            else:
                print("\nStarting fresh training.")
        else:
            print("\nStarting fresh training (resume disabled).")
        
        # ============================================================
        # STEP 3: Start training
        # ============================================================
        self.is_running = True
        self.ctrl_c_count = 0
        self._target_num_epochs = float(num_epochs)
        
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Key: {self.checkpoint_key}")
        print(f"Epochs: {num_epochs:g} (starting from {start_epoch_for_loop + 1})")
        if extend and abs(extend_epochs) > eps:
            print(f"  (Plan adjustment: {extend_epochs:+g} epochs)")
        print(f"Total minibatches processed so far: {self.total_minibatches_processed}")
        print(f"Save checkpoint every: {self.state_manager.save_every_n_minibatches} minibatches")
        print(f"Checkpoint directory: {self.state_manager.save_dir}/")
        if self.enable_validation:
            print(f"Validation: Enabled (every {self.validate_every_n_epochs} epochs)")
        else:
            print(f"Validation: Disabled")
        print(f"{'='*60}")
        print(f"Interruption handling:")
        print(f"  Ctrl+C once  → Graceful stop (save & exit)")
        print(f"  Ctrl+C twice → Force quit (no save)")
        print(f"  Cloud preempt → Emergency save & exit")
        print(f"  Power failure → Emergency save on exit")
        print(f"{'='*60}\n")
        
        train_loss = 0.0
        try:
            epoch = int(start_epoch_for_loop)
            while float(epoch) < float(num_epochs) - eps:
                if not self.is_running:
                    break

                total_batches = self.dataloader.get_total_batches_in_epoch()
                max_batches = self._batches_for_epoch(epoch, num_epochs, total_batches)
                if max_batches <= 0:
                    break
                # If resuming mid-epoch past the new partial limit, stop
                if epoch == self.start_epoch and self.start_batch >= max_batches:
                    print(
                        f"⚠ Already at/ past partial limit for epoch {epoch + 1} "
                        f"({self.start_batch}/{max_batches})."
                    )
                    break
                    
                self.current_epoch = epoch
                avg_loss = self.train_epoch(epoch, max_batches=max_batches)
                
                if not self.is_running:
                    break
                
                train_loss = avg_loss if avg_loss is not None else 0
                val_loss = None
                epoch_fully_done = (self.current_batch >= total_batches)

                if self.enable_validation and self.val_loader is not None:
                    if skip_first_validation and epoch == start_epoch_for_loop:
                        print(
                            f"Epoch {epoch + 1}/{num_epochs:g} - Loss: {train_loss:.4f} "
                            f"(Validation skipped - epoch already complete)"
                        )
                        skip_first_validation = False
                    elif epoch_fully_done and (epoch + 1) % self.validate_every_n_epochs == 0:
                        self.current_val_loss = self.run_validation(
                            loader=self.val_loader,
                            desc=f"Validation Epoch {epoch + 1}"
                        )
                        val_loss = self.current_val_loss
                        print(
                            f"Epoch {epoch + 1}/{num_epochs:g} - Loss: {train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}"
                        )
                    else:
                        partial_tag = "" if epoch_fully_done else " (partial)"
                        print(
                            f"Epoch {epoch + 1}/{num_epochs:g} - Loss: {train_loss:.4f}"
                            f"{partial_tag}"
                        )
                else:
                    partial_tag = "" if epoch_fully_done else " (partial)"
                    print(
                        f"Epoch {epoch + 1}/{num_epochs:g} - Loss: {train_loss:.4f}"
                        f"{partial_tag}"
                    )
                
                # Update history (use integer epoch id)
                epoch_num = epoch + 1
                if epoch_num in self.history.epochs:
                    idx = self.history.epochs.index(epoch_num)
                    self.history.train_losses[idx] = train_loss
                    if val_loss is not None:
                        self.history.val_losses[idx] = val_loss
                    self.history.timestamps[idx] = time.time()
                else:
                    self.history.add_epoch(epoch_num, train_loss, val_loss)

                epoch += 1
                # After first loop iteration, resume batch only applies once
                self.start_batch = 0
                        
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
                print(f"   Key: {self.checkpoint_key}")
                print(f"   Epoch {self.current_epoch + 1}, batch {self.current_batch}")
                print(f"   Resume with: python train.py")
            else:
                # Continuous progress after loop
                try:
                    tb = self.dataloader.get_total_batches_in_epoch()
                except Exception:
                    tb = 0
                if tb > 0:
                    cont = float(self.current_epoch) + float(self.current_batch) / float(tb)
                else:
                    cont = float(self.current_epoch + 1)

                if cont >= float(num_epochs) - 1e-6 or float(self.current_epoch) + 1 >= float(num_epochs) - 1e-6:
                    final_epoch = self.current_epoch
                    self.state_manager.mark_training_finished(
                        model=self.model,
                        optimizer=self.optimizer,
                        dataloader_state=self.dataloader.get_state(),
                        epoch=final_epoch,
                        batch_idx=self.current_batch,
                        total_minibatches_processed=self.total_minibatches_processed,
                        loss=train_loss if "train_loss" in locals() else 0,
                        val_loss=self.current_val_loss,
                        history=self.history.to_dict(),
                        embed_dim=getattr(self, "embed_dim", None),
                        max_seq_len=getattr(self, "max_seq_len", None),
                    )
                    print(f"\n✓ Training complete!")
                    print(f"   Key: {self.checkpoint_key}")
                    print(f"   Total minibatches: {self.total_minibatches_processed}")
                    print(f"   Plan {num_epochs:g} epochs finished.")
                    if self.current_val_loss is not None:
                        print(f"   Final Val Loss: {self.current_val_loss:.4f}")
                    self._show_history_if_available()
                else:
                    print(f"\n⚠ Training ended at minibatch {self.total_minibatches_processed}")
                    print(f"   Key: {self.checkpoint_key}")
                    print(f"   Epoch {self.current_epoch + 1}/{num_epochs:g}")
    
    def train_epoch(self, epoch, max_batches: Optional[int] = None):
        """Train one epoch with optimized timing and reduced overhead.
        
        max_batches: optional cap (for fractional --extend plans).
        """
        self.model.train()
        
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
        if max_batches is None:
            target_batches = total_batches
        else:
            target_batches = min(int(max_batches), total_batches)
        total_samples = len(self.dataloader.current_epoch_indices)

        print(
            f"🔍 DEBUG EPOCH {epoch+1}: target_batches = {target_batches}/"
            f"{total_batches}, total_samples = {total_samples}"
        )
        print(f"🔍 DEBUG: dataset len = {len(self.dataloader.dataset)}")
            
        pbar = tqdm(total=target_batches, desc=f'Epoch {epoch+1}',
                    initial=min(resume_batch_start, target_batches),
                    mininterval=0.5,
                    miniters=5)

        if resume_batch_start > 0: pbar.refresh()
        
        batch_idx = resume_batch_start
        total_loss = 0.0
        batches_in_epoch = 0
        
        total_wait_time = 0.0
        total_load_time = 0.0
        total_compute_time = 0.0
        total_cycle_time = 0.0
        
        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        device = self.device
        dataloader = self.dataloader

        cycle_start = time.perf_counter()
        wait_start = time.time()

        batch_count = 0
        while True:

            if self.stop_requested:
                self._emergency_save()
                pbar.close()
                self.is_running = False
                return total_loss / batches_in_epoch if batches_in_epoch > 0 else 0
            
            if batch_idx >= target_batches:
                break

            result = dataloader.get_next_batch()
            if result is None:
                print(f"🔍 DEBUG: get_next_batch returned None at batch_count={batch_count}")
                break
            batch, load_time = result

            if batch is None:
                print(f"🔍 DEBUG: batch is None at batch_count={batch_count}")
                break

            wait_time = time.time() - wait_start
            total_wait_time += wait_time
            total_load_time += load_time

            batch_count += 1

            # if batch_count % 1000 == 0:
            #     print(f"🔍 DEBUG: processed {batch_count} batches")
                        
            compute_start = time.perf_counter()
            
            x, y, mean, std, shape = batch   # All normalized and ready for model input
            # x: (batch_size, F, T), y: (batch_size, output_dim), mean: (batch_size, F, 1), std: (batch_size, F, 1), shape: (batch_size, F, 4)
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mean = mean.to(device, non_blocking=True)
            std = std.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            P = model(x, mean, std, shape)
            L = loss_fn(P, x, y)
            L.backward()
            optimizer.step()
            
            total_loss += L.item()
            
            compute_time = time.perf_counter() - compute_start
            total_compute_time += compute_time
            
            cycle_time = time.perf_counter() - cycle_start
            cycle_start = time.perf_counter()
            total_cycle_time += cycle_time
            
            batch_idx += 1
            batches_in_epoch += 1
            self.total_minibatches_processed += 1

            self.current_batch = batch_idx
            samples_used = min(batch_idx * self.dataloader.batch_size, total_samples)
            
            update_cycle = 10
            if batch_idx % update_cycle == 0 or batch_idx == 1:
                avg_loss = (total_loss / batches_in_epoch) if batches_in_epoch > 0 else 0
                self.current_loss = avg_loss
                avg_wait = total_wait_time / batches_in_epoch * 1000
                avg_load = total_load_time / batches_in_epoch * 1000
                avg_comp = total_compute_time / batches_in_epoch * 1000
                avg_cycle = total_cycle_time / batches_in_epoch * 1000

                pbar.update(batch_idx - pbar.n)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'wait': f'{avg_wait:.0f}ms',
                    'load': f'{avg_load:.0f}ms',
                    'comp': f'{avg_comp:.0f}ms',
                    'cycle': f'{avg_cycle:.0f}ms',
                    'data': f'{samples_used}/{total_samples}',
                    'total': self.total_minibatches_processed
                })

            if (self.total_minibatches_processed % 
                self.state_manager.save_every_n_minibatches == 0):
                
                dataloader_state = dataloader.get_state()
                
                self.state_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    dataloader_state=dataloader_state,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    total_minibatches_processed=self.total_minibatches_processed,
                    loss=avg_loss,
                    val_loss=self.current_val_loss,
                    history=self.history.to_dict(),
                    is_emergency=False,
                    training_finished=False,
                    embed_dim=getattr(self, 'embed_dim', None),
                    max_seq_len=getattr(self, 'max_seq_len', None),
                )
            wait_start = time.time()

        print(f"🔍 DEBUG: Epoch {epoch+1} completed. Processed {batch_count}/{total_batches} batches")
        
        pbar.close()

        self.dataloader._stop_prefetch()
        
        return (total_loss / batches_in_epoch) if batches_in_epoch > 0 else 0


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINER MODULE DEMO")
    print("=" * 60)
    
    class DummyModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x, mean=None, std=None, shape=None):
            if x.dim() == 3:
                x = x.reshape(x.size(0), -1)
            return self.fc(x)
    
    input_dim = 10
    output_dim = 3
    batch_size = 4
    T = 5
    F = 2
    total_samples = 100
    
    model = DummyModel(T * F, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    class DummyDataset:
        def __len__(self): return total_samples
        def __getitem__(self, idx): return torch.randn(F, T), torch.randn(output_dim), torch.randn(F), torch.randn(F), torch.randn(F, 4)
    
    class DummyLoader:
        def __init__(self, dataset):
            self.dataset = dataset
            self.batch_size = batch_size
            self.current_epoch_indices = list(range(len(dataset)))
            self.current_batch_start = 0
            self.dataset_len = len(dataset)
        def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
            self.current_batch_start = resume_batch_start
            if resume_indices is not None:
                self.current_epoch_indices = resume_indices
        def get_total_batches_in_epoch(self):
            return math.ceil(self.dataset_len / self.batch_size)
        def get_next_batch(self):
            if self.current_batch_start >= self.get_total_batches_in_epoch():
                return None, 0, 0
            remaining = self.dataset_len - self.current_batch_start * self.batch_size
            bs = min(self.batch_size, remaining)
            X_batch = torch.randn(bs, F, T)
            Y_batch = torch.randn(bs, output_dim)
            MEAN_batch = torch.randn(bs, F)
            STD_batch = torch.randn(bs, F)
            SHAPE_batch = torch.randn(bs, F, 4)
            self.current_batch_start += 1
            return (X_batch, Y_batch, MEAN_batch, STD_batch, SHAPE_batch), 0.001, 0.001
        def get_state(self):
            return {'epoch_indices': self.current_epoch_indices, 'batch_start': self.current_batch_start}
        def restore_state(self, state):
            self.current_epoch_indices = state['epoch_indices']
            self.current_batch_start = state['batch_start']
    
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    
    train_loader = DummyLoader(train_dataset)
    val_loader = DummyLoader(val_dataset)
    
    def dummy_loss(P, X, Y):
        return nn.functional.mse_loss(P, Y)
    
    trainer = ResumableTrainer(
        model=model,
        loss_fn=dummy_loss,
        optimizer=optimizer,
        dataloader=train_loader,
        save_dir='test_checkpoints',
        device='cpu',
        save_every_n_minibatches=5,
        val_loader=val_loader,
        enable_validation=True,
        validate_every_n_epochs=1,
        checkpoint_key='test_key'
    )
    
    print("\nFirst run: training for 3 epochs with validation...")
    trainer.train(num_epochs=3, resume=False)
    
    print("\n" + "=" * 60)
    print("Second run: should detect training_finished=True and show history")
    print("=" * 60)
    trainer.train(num_epochs=3, resume=True)
    
    import shutil
    if Path('test_checkpoints').exists():
        shutil.rmtree('test_checkpoints')
        print("\nCleaned up test checkpoints.")
    
    print("\nTrainer module demo complete!")