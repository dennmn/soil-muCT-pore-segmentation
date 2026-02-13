"""
Block Processor with Halo Overlap
==================================

Implements chunked processing for large 3D volumes with boundary-correct EDT.

Core Algorithm (Approved Logic):
    1. Extract padded block: volume[z0-H:z1+H, y0-H:y1+H, x0-H:x1+H]
    2. Compute EDT on entire padded block
    3. Crop halo: edt_padded[H:-H, H:-H, H:-H]
    4. Save/reassemble cropped result

Critical: Halo width H must be >= max expected pore diameter to ensure EDT accuracy.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict
from pathlib import Path
import warnings


class BlockProcessor:
    """
    Processes large 3D volumes in overlapping chunks to avoid memory overflow.
    
    Attributes:
        volume_shape: Original volume dimensions (Z, Y, X)
        chunk_size: Core block size before padding (Z, Y, X)
        halo_width: Overlap padding width in voxels
        overlap: Effective overlap between adjacent blocks (2 * halo_width)
    """
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        chunk_size: Tuple[int, int, int] = (128, 128, 128),
        halo_width: int = 50
    ):
        """
        Initialize block processor.
        
        Args:
            volume_shape: Full volume dimensions (Z, Y, X)
            chunk_size: Target chunk size for core region (before padding)
            halo_width: Padding width on each side. Must be >= max pore diameter.
                       Recommended: 2 × max_expected_pore_diameter
        
        Raises:
            ValueError: If chunk_size or halo_width invalid
        """
        self.volume_shape = volume_shape
        self.chunk_size = chunk_size
        self.halo_width = halo_width
        
        # Validation
        if halo_width < 10:
            warnings.warn(
                f"halo_width={halo_width} may be too small. "
                f"Recommend >= 2 × max_pore_diameter",
                UserWarning
            )
        
        if any(cs < 2 * halo_width for cs in chunk_size):
            raise ValueError(
                f"chunk_size {chunk_size} must be >= 2×halo_width ({2*halo_width}) "
                f"in all dimensions"
            )
        
        # Compute block grid
        self.blocks = self._compute_block_grid()
    
    def _compute_block_grid(self) -> list:
        """
        Compute all block coordinates (without halo padding).
        
        Returns:
            List of tuples: [(z0, z1, y0, y1, x0, x1), ...]
        """
        Z, Y, X = self.volume_shape
        cz, cy, cx = self.chunk_size
        
        blocks = []
        for z0 in range(0, Z, cz):
            z1 = min(z0 + cz, Z)
            for y0 in range(0, Y, cy):
                y1 = min(y0 + cy, Y)
                for x0 in range(0, X, cx):
                    x1 = min(x0 + cx, X)
                    blocks.append((z0, z1, y0, y1, x0, x1))
        
        return blocks
    
    def get_padded_slice(
        self,
        block_coords: Tuple[int, int, int, int, int, int]
    ) -> Tuple[slice, slice, slice, Tuple[int, int, int, int, int, int]]:
        """
        Get slice indices for extracting padded block from full volume.
        
        Args:
            block_coords: Core block coordinates (z0, z1, y0, y1, x0, x1)
        
        Returns:
            (z_slice, y_slice, x_slice, actual_coords):
                - Slices for extracting from full volume (clamped to boundaries)
                - Actual extracted coordinates (may be smaller than requested)
        """
        z0, z1, y0, y1, x0, x1 = block_coords
        H = self.halo_width
        Z, Y, X = self.volume_shape
        
        # Compute padded coordinates (clamped to volume boundaries)
        z0_pad = max(0, z0 - H)
        z1_pad = min(Z, z1 + H)
        y0_pad = max(0, y0 - H)
        y1_pad = min(Y, y1 + H)
        x0_pad = max(0, x0 - H)
        x1_pad = min(X, x1 + H)
        
        return (
            slice(z0_pad, z1_pad),
            slice(y0_pad, y1_pad),
            slice(x0_pad, x1_pad),
            (z0_pad, z1_pad, y0_pad, y1_pad, x0_pad, x1_pad)
        )
    
    def crop_halo(
        self,
        padded_result: np.ndarray,
        block_coords: Tuple[int, int, int, int, int, int],
        actual_padded_coords: Tuple[int, int, int, int, int, int]
    ) -> np.ndarray:
        """
        Crop halo from processed padded block to get core region.
        
        Args:
            padded_result: Result computed on padded block
            block_coords: Original core block coordinates (z0, z1, y0, y1, x0, x1)
            actual_padded_coords: Actual padded coordinates extracted
        
        Returns:
            Cropped result corresponding to core block only
        """
        z0, z1, y0, y1, x0, x1 = block_coords
        z0_pad, z1_pad, y0_pad, y1_pad, x0_pad, x1_pad = actual_padded_coords
        
        # Compute offsets within padded block
        z_start = z0 - z0_pad
        z_end = z_start + (z1 - z0)
        y_start = y0 - y0_pad
        y_end = y_start + (y1 - y0)
        x_start = x0 - x0_pad
        x_end = x_start + (x1 - x0)
        
        return padded_result[z_start:z_end, y_start:y_end, x_start:x_end]
    
    def process_volume(
        self,
        volume: np.ndarray,
        process_func: Callable[[np.ndarray], np.ndarray],
        checkpoint_manager: Optional[object] = None,
        resume: bool = False
    ) -> np.ndarray:
        """
        Process entire volume in chunks with halo overlap.
        
        Algorithm per block:
            1. Extract padded block from volume
            2. Apply process_func to padded block
            3. Crop halo from result
            4. Insert cropped result into output volume
            5. Optional: Save checkpoint
        
        Args:
            volume: Input 3D volume to process
            process_func: Function that processes a padded block.
                         Signature: func(padded_block) -> processed_block
            checkpoint_manager: Optional CheckpointManager for persistence
            resume: If True, attempt to resume from checkpoint
        
        Returns:
            Fully processed volume (same shape as input)
        
        Example:
            >>> def my_edt(block):
            ...     from scipy.ndimage import distance_transform_edt
            ...     return distance_transform_edt(block)
            >>> 
            >>> processor = BlockProcessor(volume.shape)
            >>> edt_map = processor.process_volume(volume, my_edt)
        """
        if volume.shape != self.volume_shape:
            raise ValueError(
                f"Volume shape {volume.shape} doesn't match "
                f"processor shape {self.volume_shape}"
            )
        
        # Initialize output volume
        output = np.zeros(self.volume_shape, dtype=np.float32)
        
        # Resume from checkpoint if requested
        start_idx = 0
        if resume and checkpoint_manager is not None:
            state = checkpoint_manager.load_state()
            if state is not None:
                output = state['output']
                start_idx = state['last_block_idx'] + 1
                print(f"Resuming from block {start_idx}/{len(self.blocks)}")
        
        # Process each block
        total_blocks = len(self.blocks)
        for idx, block_coords in enumerate(self.blocks[start_idx:], start=start_idx):
            print(f"Processing block {idx+1}/{total_blocks}: {block_coords}")
            
            # Step 1: Extract padded block
            z_slice, y_slice, x_slice, actual_coords = self.get_padded_slice(block_coords)
            padded_block = volume[z_slice, y_slice, x_slice]
            
            # Step 2: Process padded block
            try:
                processed_padded = process_func(padded_block)
            except Exception as e:
                raise RuntimeError(
                    f"Processing failed for block {idx}: {e}"
                ) from e
            
            # Step 3: Crop halo
            cropped_result = self.crop_halo(
                processed_padded,
                block_coords,
                actual_coords
            )
            
            # Step 4: Insert into output volume
            z0, z1, y0, y1, x0, x1 = block_coords
            output[z0:z1, y0:y1, x0:x1] = cropped_result
            
            # Step 5: Checkpoint
            if checkpoint_manager is not None:
                checkpoint_manager.save_state({
                    'output': output,
                    'last_block_idx': idx,
                    'total_blocks': total_blocks
                })
        
        return output
    
    def get_memory_estimate(self, dtype=np.float32) -> Dict[str, float]:
        """
        Estimate memory requirements for block processing.
        
        Args:
            dtype: Data type for result arrays
        
        Returns:
            Dictionary with memory estimates in MB:
                - input_block: Padded input block size
                - output_block: Padded output block size
                - total_per_block: Total memory per block iteration
                - full_output: Full output volume size
        """
        H = self.halo_width
        cz, cy, cx = self.chunk_size
        
        # Padded block size (worst case: no boundary clamping)
        padded_shape = (cz + 2*H, cy + 2*H, cx + 2*H)
        bytes_per_voxel = np.dtype(dtype).itemsize
        
        input_block_mb = np.prod(padded_shape) * 1 / (1024**2)  # Assume bool/uint8
        output_block_mb = np.prod(padded_shape) * bytes_per_voxel / (1024**2)
        
        full_output_mb = np.prod(self.volume_shape) * bytes_per_voxel / (1024**2)
        
        return {
            'input_block_mb': input_block_mb,
            'output_block_mb': output_block_mb,
            'total_per_block_mb': input_block_mb + output_block_mb,
            'full_output_mb': full_output_mb
        }


if __name__ == "__main__":
    # Validation test
    print("Block Processor - Validation Test")
    print("=" * 60)
    
    # Create test volume
    volume = np.random.rand(256, 256, 256) > 0.7
    print(f"Test volume: {volume.shape}, dtype={volume.dtype}")
    
    # Initialize processor
    processor = BlockProcessor(
        volume_shape=volume.shape,
        chunk_size=(128, 128, 128),
        halo_width=20
    )
    
    print(f"\nBlock grid: {len(processor.blocks)} blocks")
    print(f"Halo width: {processor.halo_width} voxels")
    
    # Memory estimate
    mem = processor.get_memory_estimate()
    print(f"\nMemory estimates:")
    print(f"  Per block: {mem['total_per_block_mb']:.2f} MB")
    print(f"  Full output: {mem['full_output_mb']:.2f} MB")
    
    # Test single block extraction and cropping
    block_coords = processor.blocks[0]
    z_slice, y_slice, x_slice, actual_coords = processor.get_padded_slice(block_coords)
    padded = volume[z_slice, y_slice, x_slice]
    
    print(f"\nTest block 0:")
    print(f"  Core coords: {block_coords}")
    print(f"  Padded coords: {actual_coords}")
    print(f"  Padded shape: {padded.shape}")
    
    # Simulate processing (identity function)
    cropped = processor.crop_halo(padded, block_coords, actual_coords)
    z0, z1, y0, y1, x0, x1 = block_coords
    expected_shape = (z1-z0, y1-y0, x1-x0)
    
    print(f"  Cropped shape: {cropped.shape}")
    print(f"  Expected shape: {expected_shape}")
    assert cropped.shape == expected_shape, "Crop failed!"
    
    print("\n" + "=" * 60)
    print("Module validation complete.")
