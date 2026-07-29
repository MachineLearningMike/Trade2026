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

def generate_checkpoint_key(config: dict) -> str:
    """
    Generate a unique key for checkpoints based on config parameters
    that affect training output.
    """
    key_params = {
        'patch_size': config.get('patch_size'),
        'num_layers': config.get('num_layers'),
        'num_heads': config.get('num_heads'),
        'ff_dim': config.get('ff_dim'),
        'dropout': config.get('dropout'),
        'head_hidden_dim': config.get('head_hidden_dim'),
        'causal': config.get('causal'),
        'use_reconstruction': config.get('use_reconstruction'),
        'past_days': config.get('past_days'),
        'future_minutes': config.get('future_minutes'),
        'step': config.get('step'),
        'max_accum_deviation': config.get('max_accum_deviation'),
        'min_horizon': config.get('min_horizon'),
        'max_horizon': config.get('max_horizon'),
        'normalize': config.get('normalize'),
        'horizon_mode': config.get('horizon_mode'),
        'weight_type': config.get('weight_type'),
        'weight_decay': config.get('weight_decay'),
        'slope_loss_weight': config.get('slope_loss_weight'),
        'intercept_loss_weight': config.get('intercept_loss_weight'),
        'reconstruction_loss_weight': config.get('reconstruction_loss_weight'),
        'learning_rate': config.get('learning_rate'),
        'device': config.get('device', 'cpu'),
    }
    
    param_string = json.dumps(key_params, sort_keys=True)
    key = hashlib.md5(param_string.encode()).hexdigest()[:16]
    return key


# ═══════════════════════════════════════════════════════════
# TRAINING STATE MANAGER
# ═══════════════════════════════════════════════════════════

class TrainingState:
    """Manages saving and loading of complete training state with atomic writes."""
    
    def __init__(self, save_dir='checkpoints', save_every_n_minibatches=100, 
                 max_checkpoints=5, checkpoint_key=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_minibatches = save_every_n_minibatches
        self.max_checkpoints = max_checkpoints
        
        if checkpoint_key:
            self.save_dir = self.save_dir / checkpoint_key
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.save_dir / 'training_state.json'
        self.checkpoint_key = checkpoint_key
        
    def save_checkpoint(self, model, optimizer, dataloader_state, epoch, batch_idx, 
                        total_minibatches_processed, loss, is_emergency=False, training_finished=False,
                        val_loss=None, history=None):
        """Save checkpoint with atomic write to prevent corruption."""
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
        }
        
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
                    fallback_file = self.save_dir / f'training_state_{timestamp}.json'
                    with open(fallback_file, 'w') as f:
                        json.dump(state_info, f, indent=2)
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
                               total_minibatches_processed, loss, val_loss=None, history=None):
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
            training_finished=True
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
                json.dump(state_info, f, indent=2)
            
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
    
    def get_completed_epochs(self):
        """Get the number of completed epochs from the state file."""
        if not self.state_file.exists():
            return 0
        
        try:
            with open(self.state_file, 'r') as f:
                state_info = json.load(f)
            return state_info.get('epoch', 0) + 1  # 0-indexed to 1-indexed
        except:
            return 0
        
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
        
        if state_info.get('training_finished', False):
            print(f"✓ Training is already finished!")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  Final checkpoint: {state_info['latest_checkpoint']}")
            print(f"  Total minibatches: {state_info['total_minibatches_processed']}")
            print(f"  Loss: {state_info['loss']:.4f}")
            if state_info.get('val_loss') is not None:
                print(f"  Val Loss: {state_info['val_loss']:.4f}")
            history = state_info.get('history', None)
            return 'finished', None, None, None, None, history, None, None
        
        latest_checkpoint = Path(state_info['latest_checkpoint'])
        
        if not latest_checkpoint.exists():
            print(f"⚠ Checkpoint file not found. Starting fresh.")
            return None
        
        print(f"Found checkpoint: {latest_checkpoint.name}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}. Starting fresh.")
            return None
        
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
        
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        total_minibatches_processed = checkpoint['total_minibatches_processed']
        loss = checkpoint['loss']
        val_loss = checkpoint.get('val_loss', None)
        history = checkpoint.get('history', None)
        is_emergency = checkpoint.get('is_emergency', False)
        
        if 'dataloader_state' in checkpoint:
            dataloader_state = checkpoint['dataloader_state']
            dataloader.restore_state(dataloader_state)
            
            if is_emergency:
                print(f"⚠ Resuming from EMERGENCY checkpoint")
            else:
                print(f"Resuming from checkpoint")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  Minibatch: {total_minibatches_processed}")
            print(f"  Epoch {epoch+1}, batch {batch_idx}")
            print(f"  Loss: {loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            
            return epoch, batch_idx, total_minibatches_processed, loss, val_loss, history, dataloader_state, is_emergency
        
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
    
    def __init__(self, model, loss_fn, optimizer, dataloader, 
                 save_dir='checkpoints', device='cpu',
                 save_every_n_minibatches=100,
                 val_loader=None, 
                 enable_validation=False, 
                 validate_every_n_epochs=1,
                 checkpoint_key=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        
        self.enable_validation = enable_validation
        self.validate_every_n_epochs = validate_every_n_epochs
        
        # Generate checkpoint key if not provided
        if checkpoint_key is None:
            config = get_config()
            checkpoint_key = generate_checkpoint_key(config)
        
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
                training_finished=False
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
            pbar = tqdm(total=total_batches, desc=desc, mininterval=0.5, miniters=5)
            while True:
                batch, load_time = loader.get_next_batch()
                if batch is None:
                    break
                
                x, y, mean, std, shape = batch
                x, y = x.to(self.device), y.to(self.device)
                x = x.transpose(1, 2)
                
                P = self.model(x, mean, std, shape)
                L = self.loss_fn(P, x, y)
                
                total_loss += L.item()
                num_batches += 1
                
                update_cycle = 5
                if num_batches % update_cycle == 0 or num_batches == 1:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                if num_batches % update_cycle == 0:
                    pbar.update(update_cycle if num_batches > update_cycle else 1)
                    pbar.set_postfix({'loss': f'{L.item():.4f}'})
            
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
    
    def train(self, num_epochs, resume=True, extend_epochs=0):
        """Train the model with robust checkpointing.
        
        Args:
            num_epochs: Base number of epochs to train
            resume: Whether to resume from checkpoint
            extend_epochs: Number of additional epochs to add (0 = no extension)
        """
        
        # If extending, calculate the new total epochs based on completed epochs
        if extend_epochs > 0:
            # Clear the finished flag
            self.state_manager.clear_finished_flag()
            
            # Get the actual completed epochs from the state file
            completed_epochs = self.state_manager.get_completed_epochs()
            
            # Calculate new total
            new_total = completed_epochs + extend_epochs
            num_epochs = new_total
            
            print(f"\n{'='*60}")
            print(f"⚠ EXTENDING TRAINING BY {extend_epochs} EPOCHS")
            print(f"{'='*60}")
            print(f"  Key: {self.checkpoint_key}")
            print(f"  Completed epochs: {completed_epochs}")
            print(f"  Adding: {extend_epochs} epochs")
            print(f"  New total epochs: {new_total}")
            print(f"{'='*60}\n")
        
        # Initialize
        training_complete = False
        skip_first_validation = False
        start_epoch_for_loop = 0
        
        if resume:
            loaded = self.state_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.dataloader
            )
            
            if loaded:
                if loaded == 'finished':
                    if extend_epochs > 0:
                        print("⚠ training_finished flag still set. Retrying clear...")
                        self.state_manager.clear_finished_flag()
                        loaded = self.state_manager.load_latest_checkpoint(
                            self.model, self.optimizer, self.dataloader
                        )
                        if loaded == 'finished':
                            print("❌ Failed to clear training_finished flag. Aborting.")
                            return
                        (self.start_epoch, self.start_batch, 
                         self.total_minibatches_processed, _, val_loss, history_data, _, is_emergency) = loaded
                        self.current_val_loss = val_loss
                    else:
                        print("\n" + "=" * 60)
                        print("✓ TRAINING ALREADY FINISHED")
                        print("=" * 60)
                        print(f"Key: {self.checkpoint_key}")
                        print(f"All previous epochs completed.")
                        print("Nothing to train.")
                        print("Use --clean to start fresh, or --extend N to add N more epochs.")
                        print("=" * 60)
                        try:
                            with open(self.state_manager.state_file, 'r') as f:
                                state_info = json.load(f)
                            history_data = state_info.get('history', None)
                            if history_data:
                                self.history.from_dict(history_data)
                                self._show_history_if_available()
                        except:
                            pass
                        return
                
                # Unpack checkpoint
                (self.start_epoch, self.start_batch, 
                 self.total_minibatches_processed, _, val_loss, history_data, _, is_emergency) = loaded
                self.current_val_loss = val_loss
                
                if history_data:
                    self.history.from_dict(history_data)
                    print(f"✓ Loaded training history ({len(self.history.epochs)} epochs)")
                
                total_batches_in_epoch = self.dataloader.get_total_batches_in_epoch()
                
                # Check if training is actually complete
                training_complete = False
                if self.start_epoch >= num_epochs - 1:
                    if self.start_epoch > num_epochs - 1:
                        training_complete = True
                    elif self.start_epoch == num_epochs - 1 and self.start_batch >= total_batches_in_epoch:
                        training_complete = True
                
                if training_complete:
                    if extend_epochs > 0:
                        print(f"\n⚠ WARNING: Training already complete at epoch {self.start_epoch+1}")
                        print(f"  Key: {self.checkpoint_key}")
                        print(f"  --extend={extend_epochs} was not enough.")
                        print(f"  Use a larger --extend value.")
                        return
                    
                    self.state_manager.mark_training_finished(
                        model=self.model,
                        optimizer=self.optimizer,
                        dataloader_state=self.dataloader.get_state(),
                        epoch=self.start_epoch,
                        batch_idx=self.start_batch,
                        total_minibatches_processed=self.total_minibatches_processed,
                        loss=self.current_loss if self.current_loss != 0 else 0,
                        val_loss=self.current_val_loss,
                        history=self.history.to_dict()
                    )
                    print("\n" + "=" * 60)
                    print("✓ TRAINING ALREADY FINISHED")
                    print("=" * 60)
                    print(f"Key: {self.checkpoint_key}")
                    print(f"All {num_epochs} epochs completed in previous run.")
                    print("Nothing to train.")
                    print("Use --clean to start fresh, or --extend N to add N more epochs.")
                    print("=" * 60)
                    self._show_history_if_available()
                    return
                
                # Determine where to start the training loop
                if self.start_batch >= total_batches_in_epoch:
                    # The epoch is fully completed
                    if extend_epochs > 0:
                        # Extending - don't skip validation
                        start_epoch_for_loop = self.start_epoch + 1
                        skip_first_validation = False
                        print(f"⚠ Extending: starting from epoch {self.start_epoch+2}")
                    else:
                        # Normal resume from completed epoch - skip validation
                        start_epoch_for_loop = self.start_epoch + 1
                        skip_first_validation = True
                        print(f"⚠ Resuming from completed epoch {self.start_epoch+1}")
                        print(f"  Skipping validation for epoch {self.start_epoch+1}")
                else:
                    # Partial epoch - resume normally
                    start_epoch_for_loop = self.start_epoch
                    skip_first_validation = False
                
                if is_emergency:
                    print(f"\n⚠ Resuming from emergency save.")
                    print(f"  Batch {self.start_batch} was completed before save.")
                    print(f"  Starting from batch {self.start_batch}.")
                    
                    # For emergency resume, we need to start from the current epoch
                    # with the correct batch index
                    start_epoch_for_loop = self.start_epoch
                    # The train_epoch will handle the batch resumption via start_batch
            else:
                print("\nStarting fresh training.")
        else:
            print("\nStarting fresh training (resume disabled).")
        
        self.is_running = True
        self.ctrl_c_count = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Key: {self.checkpoint_key}")
        print(f"Epochs: {num_epochs} (starting from {start_epoch_for_loop+1})")
        if extend_epochs > 0:
            print(f"  (Extension mode: continuing from checkpoint)")
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
        
        try:
            for epoch in range(start_epoch_for_loop, num_epochs):
                if not self.is_running:
                    break
                    
                self.current_epoch = epoch
                avg_loss = self.train_epoch(epoch)
                
                if not self.is_running:
                    break
                
                train_loss = avg_loss if avg_loss is not None else 0
                val_loss = None
                
                if self.enable_validation and self.val_loader is not None:
                    if skip_first_validation and epoch == start_epoch_for_loop:
                        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} (Validation skipped - epoch already complete)')
                        skip_first_validation = False
                    elif (epoch + 1) % self.validate_every_n_epochs == 0:
                        self.current_val_loss = self.run_validation(
                            loader=self.val_loader,
                            desc=f"Validation Epoch {epoch+1}"
                        )
                        val_loss = self.current_val_loss
                        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    else:
                        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}')
                
                # Only add to history if this epoch is new (not already in history)
                epoch_num = epoch + 1
                if epoch_num not in self.history.epochs:
                    self.history.add_epoch(epoch_num, train_loss, val_loss)
                else:
                    # Update existing epoch's loss if it was previously recorded with placeholder
                    idx = self.history.epochs.index(epoch_num)
                    self.history.train_losses[idx] = train_loss
                    if val_loss is not None:
                        self.history.val_losses[idx] = val_loss
                    self.history.timestamps[idx] = time.time()
                    
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
                print(f"   Epoch {self.current_epoch+1}, batch {self.current_batch}")
                print(f"   Resume with: python train.py")
            elif self.current_epoch + 1 >= num_epochs:
                self.state_manager.mark_training_finished(
                    model=self.model,
                    optimizer=self.optimizer,
                    dataloader_state=self.dataloader.get_state(),
                    epoch=self.current_epoch,
                    batch_idx=self.current_batch,
                    total_minibatches_processed=self.total_minibatches_processed,
                    loss=train_loss if 'train_loss' in locals() else 0,
                    val_loss=self.current_val_loss,
                    history=self.history.to_dict()
                )
                print(f"\n✓ Training complete!")
                print(f"   Key: {self.checkpoint_key}")
                print(f"   Total minibatches: {self.total_minibatches_processed}")
                print(f"   All {num_epochs} epochs finished.")
                if self.current_val_loss is not None:
                    print(f"   Final Val Loss: {self.current_val_loss:.4f}")
                self._show_history_if_available()
            else:
                print(f"\n⚠ Training ended unexpectedly at minibatch {self.total_minibatches_processed}")
                print(f"   Key: {self.checkpoint_key}")
                print(f"   Epoch {self.current_epoch+1}/{num_epochs}")
    
    def train_epoch(self, epoch):
        """Train one epoch with optimized timing and reduced overhead."""
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
        total_samples = len(self.dataloader.current_epoch_indices)

        print(f"🔍 DEBUG EPOCH {epoch+1}: total_batches = {total_batches}, total_samples = {total_samples}")
        print(f"🔍 DEBUG: dataset len = {len(self.dataloader.dataset)}")
            
        # Optimized tqdm - update less frequently
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch+1}',
                    initial=resume_batch_start,
                    mininterval=0.5,
                    miniters=5)
        
        batch_idx = resume_batch_start
        total_loss = 0.0
        batches_in_epoch = 0
        
        # Track timing
        total_wait_time = 0.0
        total_load_time = 0.0
        total_compute_time = 0.0
        total_cycle_time = 0.0
        
        # Cache local references for speed
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
            
            # Get batch
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

            if batch_count % 1000 == 0:
                print(f"🔍 DEBUG: processed {batch_count} batches")
                        
            compute_start = time.perf_counter()
            
            x, y, mean, std, shape = batch
            
            # Transfer to device with non_blocking
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mean = mean.to(device, non_blocking=True)
            std = std.to(device, non_blocking=True)
            shape = shape.to(device, non_blocking=True)
            
            # Transpose
            x = x.transpose(1, 2)
            
            # Forward + Backward + Optimizer
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
            
            # Update metrics
            batch_idx += 1
            batches_in_epoch += 1
            self.total_minibatches_processed += 1

            self.current_batch = batch_idx
            samples_used = min(batch_idx * self.dataloader.batch_size, total_samples)
            
            # Update progress bar less frequently (every 10 batches)
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

            # Checkpoint check
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
                    training_finished=False
                )
            wait_start = time.time()

        print(f"🔍 DEBUG: Epoch {epoch+1} completed. Processed {batch_count}/{total_batches} batches")
        
        # Final pbar update
        pbar.close()

        # Cleanup at end of epoch
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