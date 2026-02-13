"""
Checkpoint Manager for Colab Timeout Resilience
================================================

Provides persistence for block processing to handle Colab runtime disconnections.

Features:
    - Save/load processing state to Google Drive
    - Resume from last completed block
    - Automatic periodic checkpointing
    - Compressed numpy arrays for storage efficiency
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import pickle
import gzip
import json
import warnings
from datetime import datetime

try:
    from .config_loader import load_config
except ImportError:
    from config_loader import load_config

_CONFIG = load_config()
_PATH_CONFIG = _CONFIG.get("paths", {})
_PROCESSING_CONFIG = _CONFIG.get("processing_thresholds", {})
DEFAULT_CHECKPOINT_DIR = _PATH_CONFIG.get("checkpoint_dir")
DEFAULT_CHECKPOINT_COMPRESSION = _PROCESSING_CONFIG.get("checkpoint_compression", True)
DEFAULT_CHECKPOINT_AUTO_BACKUP = _PROCESSING_CONFIG.get("checkpoint_auto_backup", True)


class CheckpointManager:
    """
    Manages checkpointing for long-running block processing jobs.
    
    Attributes:
        checkpoint_dir: Directory for storing checkpoint files
        run_id: Unique identifier for this processing run
        compress: Whether to compress checkpoint data
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        compress: Optional[bool] = None,
        auto_backup: Optional[bool] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., /content/drive/MyDrive/checkpoints)
            run_id: Unique run identifier. If None, generates timestamp-based ID.
            compress: If True, use gzip compression for large arrays
                      (defaults from pores_analysis/config.yaml)
            auto_backup: If True, keep previous checkpoint as backup
                         (defaults from pores_analysis/config.yaml)
        
        Example:
            >>> manager = CheckpointManager(
            ...     checkpoint_dir="/content/drive/MyDrive/psd_checkpoints",
            ...     run_id="scan_001_psd"
            ... )
        """
        checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        if checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided via config or constructor"
            )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id is None:
            run_id = f"psd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_id = run_id
        
        self.compress = DEFAULT_CHECKPOINT_COMPRESSION if compress is None else compress
        self.auto_backup = DEFAULT_CHECKPOINT_AUTO_BACKUP if auto_backup is None else auto_backup
        
        # File paths
        self.state_file = self.checkpoint_dir / f"{run_id}_state.pkl.gz"
        self.metadata_file = self.checkpoint_dir / f"{run_id}_metadata.json"
        self.backup_file = self.checkpoint_dir / f"{run_id}_state_backup.pkl.gz"
        
        print(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        print(f"  Run ID: {run_id}")
    
    def save_state(
        self,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save processing state to disk.
        
        Args:
            state: Dictionary containing processing state.
                  Must include:
                      - 'output': np.ndarray - Current output volume
                      - 'last_block_idx': int - Index of last completed block
                      - 'total_blocks': int - Total number of blocks
            metadata: Optional metadata (config, parameters, etc.)
        
        Raises:
            ValueError: If required keys missing from state
            IOError: If save fails
        """
        # Validate state
        required_keys = {'output', 'last_block_idx', 'total_blocks'}
        if not required_keys.issubset(state.keys()):
            raise ValueError(
                f"State must contain keys: {required_keys}, got {state.keys()}"
            )
        
        # Backup existing checkpoint
        if self.auto_backup and self.state_file.exists():
            try:
                self.state_file.rename(self.backup_file)
            except Exception as e:
                warnings.warn(f"Failed to create backup: {e}", UserWarning)
        
        # Save state
        try:
            if self.compress:
                with gzip.open(self.state_file, 'wb') as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(self.state_file, 'wb') as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Failed to save checkpoint: {e}") from e
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'run_id': self.run_id,
            'last_updated': datetime.now().isoformat(),
            'last_block_idx': state['last_block_idx'],
            'total_blocks': state['total_blocks'],
            'progress_pct': 100 * (state['last_block_idx'] + 1) / state['total_blocks']
        })
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save metadata: {e}", UserWarning)
        
        # Progress message
        pct = metadata['progress_pct']
        print(f"Checkpoint saved: Block {state['last_block_idx']+1}/{state['total_blocks']} ({pct:.1f}%)")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load processing state from disk.
        
        Returns:
            State dictionary if checkpoint exists, None otherwise.
            Contains: 'output', 'last_block_idx', 'total_blocks'
        
        Raises:
            IOError: If checkpoint exists but cannot be loaded
        """
        if not self.state_file.exists():
            print("No checkpoint found, starting from scratch.")
            return None
        
        try:
            if self.compress:
                with gzip.open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
            else:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
            
            # Validate loaded state
            required_keys = {'output', 'last_block_idx', 'total_blocks'}
            if not required_keys.issubset(state.keys()):
                raise ValueError(
                    f"Corrupted checkpoint: missing keys {required_keys - state.keys()}"
                )
            
            print(f"Checkpoint loaded: Block {state['last_block_idx']+1}/{state['total_blocks']}")
            return state
            
        except Exception as e:
            # Try backup
            if self.backup_file.exists():
                warnings.warn(
                    f"Primary checkpoint corrupted ({e}), trying backup",
                    UserWarning
                )
                try:
                    if self.compress:
                        with gzip.open(self.backup_file, 'rb') as f:
                            state = pickle.load(f)
                    else:
                        with open(self.backup_file, 'rb') as f:
                            state = pickle.load(f)
                    print("Backup checkpoint loaded successfully")
                    return state
                except Exception as e2:
                    raise IOError(
                        f"Both primary and backup checkpoints failed: {e}, {e2}"
                    ) from e2
            else:
                raise IOError(f"Failed to load checkpoint: {e}") from e
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint metadata.
        
        Returns:
            Metadata dictionary if exists, None otherwise
        """
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load metadata: {e}", UserWarning)
            return None
    
    def clear_checkpoints(self) -> None:
        """
        Delete all checkpoint files for this run.
        
        Use after successful completion or to restart from scratch.
        """
        files_to_remove = [
            self.state_file,
            self.metadata_file,
            self.backup_file
        ]
        
        for file in files_to_remove:
            if file.exists():
                try:
                    file.unlink()
                    print(f"Deleted: {file.name}")
                except Exception as e:
                    warnings.warn(f"Failed to delete {file.name}: {e}", UserWarning)
    
    def get_checkpoint_size(self) -> Dict[str, float]:
        """
        Get checkpoint file sizes in MB.
        
        Returns:
            Dictionary with file sizes
        """
        sizes = {}
        
        if self.state_file.exists():
            sizes['state_mb'] = self.state_file.stat().st_size / (1024**2)
        
        if self.metadata_file.exists():
            sizes['metadata_kb'] = self.metadata_file.stat().st_size / 1024
        
        if self.backup_file.exists():
            sizes['backup_mb'] = self.backup_file.stat().st_size / (1024**2)
        
        return sizes


if __name__ == "__main__":
    # Validation test
    import tempfile
    
    print("Checkpoint Manager - Validation Test")
    print("=" * 60)
    
    # Create temporary checkpoint directory
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            run_id="test_run"
        )
        
        # Simulate processing state
        test_volume = np.random.rand(100, 100, 100).astype(np.float32)
        state = {
            'output': test_volume,
            'last_block_idx': 5,
            'total_blocks': 10
        }
        
        metadata = {
            'voxel_spacing': (2.0, 1.0, 1.0),
            'chunk_size': (64, 64, 64),
            'halo_width': 20
        }
        
        print("\n[Saving checkpoint...]")
        manager.save_state(state, metadata)
        
        sizes = manager.get_checkpoint_size()
        print(f"Checkpoint size: {sizes.get('state_mb', 0):.2f} MB")
        
        print("\n[Loading checkpoint...]")
        loaded_state = manager.load_state()
        
        assert loaded_state is not None, "Failed to load checkpoint"
        assert np.array_equal(loaded_state['output'], test_volume), "Data mismatch"
        assert loaded_state['last_block_idx'] == 5, "Block index mismatch"
        print("Checkpoint data verified ✓")
        
        print("\n[Loading metadata...]")
        loaded_metadata = manager.load_metadata()
        assert loaded_metadata is not None, "Failed to load metadata"
        print(f"Progress: {loaded_metadata['progress_pct']:.1f}%")
        
        print("\n[Cleaning up...]")
        manager.clear_checkpoints()
    
    print("\n" + "=" * 60)
    print("Module validation complete.")
