import importlib
import importlib.util
import numpy as np
from scipy.ndimage import label, convolve
from typing import Dict

# Class value constants
PORE_CLASS = 0
SOLID_CLASS = 2

def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _get_torch():
    if not _torch_available():
        return None
    return importlib.import_module("torch")


def _configure_torch_determinism(torch_module) -> None:
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    try:
        torch_module.use_deterministic_algorithms(True)
    except Exception as exc:
        raise RuntimeError(
            "Deterministic algorithms are required for GPU metrics but could not be enabled."
        ) from exc


def _compute_z_metrics_cpu(volume: np.ndarray, target_class: int, window_size: int) -> Dict[str, np.ndarray]:
    """
    Compute Z-axis stability metrics including frequency, flip rate, and class fractions.
    
    Args:
        volume: 3D volume (Z, Y, X)
        target_class: Target class to analyze
        window_size: Window size for local Z analysis
    
    Returns:
        Dictionary with metric arrays
    """
    Z, Y, X = volume.shape
    half_window = window_size // 2
    frequency = np.zeros((Z, Y, X), dtype=np.float32)
    flip_rate = np.zeros((Z, Y, X), dtype=np.int32)
    class_0_fraction = np.zeros((Z, Y, X), dtype=np.float32)
    class_2_fraction = np.zeros((Z, Y, X), dtype=np.float32)
    effective_window_size = np.zeros((Z, Y, X), dtype=np.int32)

    for z in range(Z):
        z_start = max(0, z - half_window)
        z_end = min(Z, z + half_window + 1)
        window = volume[z_start:z_end, :, :]
        window_len = z_end - z_start
        effective_window_size[z, :, :] = window_len
        frequency[z] = np.sum(window == target_class, axis=0).astype(np.float32) / window_len
        class_0_fraction[z] = np.sum(window == PORE_CLASS, axis=0).astype(np.float32) / window_len
        class_2_fraction[z] = np.sum(window == SOLID_CLASS, axis=0).astype(np.float32) / window_len
        if window_len > 1:
            transitions = np.diff(window, axis=0) != 0
            flip_rate[z] = np.sum(transitions, axis=0).astype(np.int32)
            
    return {'frequency': frequency, 'flip_rate': flip_rate, 'class_0_fraction': class_0_fraction, 
            'class_2_fraction': class_2_fraction, 'effective_window_size': effective_window_size}


def _compute_z_metrics_gpu(volume: np.ndarray, target_class: int, window_size: int) -> Dict[str, np.ndarray]:
    
    torch = _get_torch()
    if torch is None:
        raise RuntimeError("Torch is not available for GPU metrics.")
    _configure_torch_determinism(torch)
    device = torch.device("cuda")

    volume_t = torch.as_tensor(volume, device=device)
    Z, Y, X = volume_t.shape
    half_window = window_size // 2

    z_idx = torch.arange(Z, device=device, dtype=torch.long)
    z_start = torch.clamp(z_idx - half_window, min=0)
    z_end = torch.clamp(z_idx + half_window + 1, max=Z)
    window_len = (z_end - z_start).to(torch.int32)
    window_len_f = window_len.to(torch.float32).view(Z, 1, 1)
    window_len_full = window_len.view(Z, 1, 1).expand(Z, Y, X)

    def window_sum(indicator: "torch.Tensor") -> "torch.Tensor":
        indicator_int = indicator.to(torch.int32)
        cumsum = torch.cumsum(indicator_int, dim=0)
        padded = torch.cat(
            [torch.zeros((1, Y, X), device=device, dtype=cumsum.dtype), cumsum],
            dim=0,
        )
        return padded[z_end] - padded[z_start]

    target_counts = window_sum(volume_t == target_class)
    class_0_counts = window_sum(volume_t == PORE_CLASS)
    class_2_counts = window_sum(volume_t == SOLID_CLASS)

    frequency = target_counts.to(torch.float32) / window_len_f
    class_0_fraction = class_0_counts.to(torch.float32) / window_len_f
    class_2_fraction = class_2_counts.to(torch.float32) / window_len_f

    transitions = volume_t[1:] != volume_t[:-1]
    trans_int = transitions.to(torch.int32)
    trans_cumsum = torch.cumsum(trans_int, dim=0)
    trans_padded = torch.cat(
        [torch.zeros((1, Y, X), device=device, dtype=trans_cumsum.dtype), trans_cumsum],
        dim=0,
    )
    z_end_minus1 = (z_end - 1).to(torch.long)
    flip_rate = (trans_padded[z_end_minus1] - trans_padded[z_start]).to(torch.int32)

    return {
        'frequency': frequency.cpu().numpy(),
        'flip_rate': flip_rate.cpu().numpy(),
        'class_0_fraction': class_0_fraction.cpu().numpy(),
        'class_2_fraction': class_2_fraction.cpu().numpy(),
        'effective_window_size': window_len_full.cpu().numpy(),
    }


def compute_z_metrics(
    volume: np.ndarray,
    target_class: int,
    window_size: int,
    use_gpu: bool = False,
    validate_gpu: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute Z-axis stability metrics including frequency, flip rate, and class fractions.

    Args:
        volume: 3D volume (Z, Y, X)
        target_class: Target class to analyze
        window_size: Window size for local Z analysis
        use_gpu: If True and CUDA is available, compute on GPU via PyTorch
        validate_gpu: If True, assert GPU outputs match CPU outputs exactly

    Returns:
        Dictionary with metric arrays
    """
    if use_gpu and _torch_available():
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            gpu_result = _compute_z_metrics_gpu(volume, target_class, window_size)
            if validate_gpu:
                cpu_result = _compute_z_metrics_cpu(volume, target_class, window_size)
                for key, cpu_value in cpu_result.items():
                    gpu_value = gpu_result[key]
                    if np.array_equal(cpu_value, gpu_value):
                        continue
                    if cpu_value.dtype.kind in ("f", "c"):
                        diff = np.abs(cpu_value - gpu_value)
                        print(
                            f"GPU mismatch for {key}: "
                            f"max_abs_diff={np.max(diff):.8f}, "
                            f"mean_abs_diff={np.mean(diff):.8f}"
                        )
                        raise AssertionError(f"GPU mismatch for {key}.")
                    raise AssertionError(f"GPU mismatch for {key}.")
            return gpu_result
    return _compute_z_metrics_cpu(volume, target_class, window_size)

def compute_2d_metrics(slice_mask: np.ndarray, target_class: int) -> Dict[str, np.ndarray]:
    """
    Compute 2D spatial metrics for a single slice.
    
    Args:
        slice_mask: 2D mask slice (Y, X)
        target_class: Target class to analyze
    
    Returns:
        Dictionary with metric arrays
    """
    Y, X = slice_mask.shape
    class_0_neighbors = np.zeros((Y, X), dtype=np.float32)
    class_2_neighbors = np.zeros((Y, X), dtype=np.float32)
    component_size = np.zeros((Y, X), dtype=np.int32)
    
    target_mask = (slice_mask == target_class)
    if not np.any(target_mask):
        return {'class_0_neighbors': class_0_neighbors, 'class_2_neighbors': class_2_neighbors, 'component_size': component_size}

    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(target_mask, structure=structure)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        component_size = component_sizes[labeled]

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    total_neighbors = convolve(np.ones_like(slice_mask, dtype=np.float32), kernel, mode='constant', cval=0)
    
    class_0_neighbor_counts = convolve((slice_mask == PORE_CLASS).astype(np.float32), kernel, mode='constant', cval=0)
    class_2_neighbor_counts = convolve((slice_mask == SOLID_CLASS).astype(np.float32), kernel, mode='constant', cval=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        c0 = np.nan_to_num(class_0_neighbor_counts / total_neighbors)
        c2 = np.nan_to_num(class_2_neighbor_counts / total_neighbors)

    class_0_neighbors = np.where(target_mask, c0, 0.0)
    class_2_neighbors = np.where(target_mask, c2, 0.0)
    
    return {'class_0_neighbors': class_0_neighbors, 'class_2_neighbors': class_2_neighbors, 'component_size': component_size}

def compute_2d_metrics_stack(volume: np.ndarray, target_class: int) -> Dict[str, np.ndarray]:
    """
    Compute 2D spatial metrics for all slices in a volume.
    
    Args:
        volume: 3D volume (Z, Y, X)
        target_class: Target class to analyze
    
    Returns:
        Dictionary with 3D metric arrays
    """
    Z = volume.shape[0]
    first = compute_2d_metrics(volume[0], target_class)
    metrics_3d = {k: np.zeros((Z,) + volume.shape[1:], dtype=v.dtype) for k, v in first.items()}
    for z in range(Z):
        res = compute_2d_metrics(volume[z], target_class)
        for k, v in res.items(): 
            metrics_3d[k][z] = v
    return metrics_3d

def _compute_adjacent_slice_consistency_cpu(mask: np.ndarray, target_class: int) -> np.ndarray:
    """
    Compute Dice coefficient between adjacent slices for the target class.
    
    Args:
        mask: 3D mask volume (Z, Y, X)
        target_class: Target class to analyze
    
    Returns:
        1D array of Dice scores for each adjacent slice pair (length Z-1)
    """
    Z = mask.shape[0]
    dice_scores = np.zeros(Z - 1, dtype=np.float32)
    
    for z in range(Z - 1):
        slice_a = (mask[z] == target_class)
        slice_b = (mask[z + 1] == target_class)
        
        intersection = np.sum(slice_a & slice_b)
        union = np.sum(slice_a) + np.sum(slice_b)
        
        if union > 0:
            dice_scores[z] = 2.0 * intersection / union
        else:
            dice_scores[z] = 1.0  # Both empty, perfect agreement
    
    return dice_scores


def _compute_adjacent_slice_consistency_gpu(mask: np.ndarray, target_class: int) -> np.ndarray:
    torch = _get_torch()
    if torch is None:
        raise RuntimeError("Torch is not available for GPU metrics.")
    _configure_torch_determinism(torch)
    device = torch.device("cuda")

    mask_t = torch.as_tensor(mask, device=device)
    target_mask = mask_t == target_class
    slice_a = target_mask[:-1]
    slice_b = target_mask[1:]

    intersection = (slice_a & slice_b).sum(dim=(1, 2))
    union = slice_a.sum(dim=(1, 2)) + slice_b.sum(dim=(1, 2))

    intersection_f = intersection.to(torch.float64)
    union_f = union.to(torch.float64)
    ones = torch.ones_like(union_f)
    dice = torch.where(union_f > 0, (2.0 * intersection_f) / union_f, ones)

    return dice.to(torch.float32).cpu().numpy()


def compute_adjacent_slice_consistency(
    mask: np.ndarray,
    target_class: int,
    use_gpu: bool = False,
    validate_gpu: bool = False,
) -> np.ndarray:
    """
    Compute Dice coefficient between adjacent slices for the target class.

    Args:
        mask: 3D mask volume (Z, Y, X)
        target_class: Target class to analyze
        use_gpu: If True and CUDA is available, compute on GPU via PyTorch
        validate_gpu: If True, assert GPU outputs match CPU outputs exactly

    Returns:
        1D array of Dice scores for each adjacent slice pair (length Z-1)
    """
    if use_gpu and _torch_available():
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            gpu_result = _compute_adjacent_slice_consistency_gpu(mask, target_class)
            if validate_gpu:
                cpu_result = _compute_adjacent_slice_consistency_cpu(mask, target_class)
                if not np.array_equal(cpu_result, gpu_result):
                    diff = np.abs(cpu_result - gpu_result)
                    print(
                        "GPU mismatch for adjacent slice consistency: "
                        f"max_abs_diff={np.max(diff):.8f}, "
                        f"mean_abs_diff={np.mean(diff):.8f}"
                    )
                    raise AssertionError("GPU mismatch for adjacent slice consistency.")
            return gpu_result
    return _compute_adjacent_slice_consistency_cpu(mask, target_class)

def compute_z_run_length_stability(mask: np.ndarray, target_class: int) -> Dict[str, float]:
    """
    Compute Z-axis run-length stability: longest continuous run per voxel.
    
    Args:
        mask: 3D mask volume (Z, Y, X)
        target_class: Target class to analyze
    
    Returns:
        Dictionary with 'run_length_map' (Y, X) and statistics
    """
    Z, Y, X = mask.shape
    max_run_length = np.zeros((Y, X), dtype=np.int32)
    
    target_volume = (mask == target_class).astype(np.int32)
    
    for y in range(Y):
        for x in range(X):
            z_line = target_volume[:, y, x]
            current_run = 0
            max_run = 0
            for val in z_line:
                if val == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_run_length[y, x] = max_run
    
    runs = max_run_length[max_run_length > 0]
    mean_run = float(np.mean(runs)) if len(runs) > 0 else 0.0
    median_run = float(np.median(runs)) if len(runs) > 0 else 0.0
    
    return {
        'run_length_map': max_run_length,
        'mean_run_length': mean_run,
        'median_run_length': median_run
    }

def compute_component_persistence(mask: np.ndarray, target_class: int) -> Dict[str, float]:
    """
    Track connected components slice-to-slice and measure persistence.
    
    Persistence Definition:
    -----------------------
    - A component's persistence = total number of consecutive Z-slices it appears in
    - Minimum persistence = 2 (component must appear in at least 2 consecutive slices to be counted)
    - Single-slice components contribute 0 to statistics (not counted as persistent)
    
    Empty Slice Handling:
    ---------------------
    - If a slice contains no target_class voxels, tracking resets completely
    - Components before and after empty slices are treated as separate entities
    - This prevents false continuity across gaps
    
    Merge/Split Handling:
    ---------------------
    - Uses greedy "best overlap" matching between adjacent slices
    - If multiple components merge into one:
      * Only the component with highest pixel overlap is tracked as "continuing"
      * Other components are considered to have ended
    - If one component splits into multiple:
      * Each new component competes independently for overlap with parent
      * Only the one with highest overlap inherits the persistence count
    - This is a simplified heuristic; complex topological changes may not be tracked accurately
    
    Args:
        mask: 3D mask volume (Z, Y, X)
        target_class: Target class to analyze
    
    Returns:
        Dictionary with persistence statistics:
        - mean_persistence: Average lifespan of all tracked components
        - median_persistence: Median lifespan of all tracked components
        - max_persistence: Longest lifespan observed
        Returns 0.0/0.0/0 if no components persist for ≥2 slices
    """
    Z = mask.shape[0]
    # 8-connectivity for 2D component labeling
    structure = np.ones((3, 3), dtype=int)
    
    # Track components through slices
    prev_labeled = None
    prev_num = 0
    # Maps component_id -> total number of consecutive slices
    component_lifespans = {}  # Maps prev_id -> lifespan count
    next_component_id = 0
    
    for z in range(Z):
        slice_mask = (mask[z] == target_class)
        if not np.any(slice_mask):
            # Empty slice handling: reset tracking
            prev_labeled = None
            prev_num = 0
            continue
        
        curr_labeled, curr_num = label(slice_mask, structure=structure)
        
        if prev_labeled is not None:
            # Match components between current and previous slice
            matched_prev = set()
            # Greedy best-overlap matching
            curr_to_prev = {}  # Maps current ID to previous ID
            
            # Match components between slices by overlap
            for curr_id in range(1, curr_num + 1):
                curr_component = (curr_labeled == curr_id)
                best_overlap = 0
                best_prev_id = None
                
                # Find previous component with maximum pixel overlap
                for prev_id in range(1, prev_num + 1):
                    prev_component = (prev_labeled == prev_id)
                    overlap = np.sum(curr_component & prev_component)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_prev_id = prev_id
                
                # Track persistence if overlap exists
                if best_overlap > 0 and best_prev_id is not None:
                    # Component continues from previous slice
                    curr_to_prev[curr_id] = best_prev_id
                    matched_prev.add(best_prev_id)
                    if best_prev_id not in component_lifespans:
                        # First continuation: count previous slice + current slice = 2
                        component_lifespans[best_prev_id] = 2  # Previous slice + current slice
                    else:
                        # Ongoing continuation: increment lifespan
                        component_lifespans[best_prev_id] += 1
                else:
                    # New component appearing - not yet counted (needs ≥2 slices)
                    pass  # Don't add yet, wait to see if it continues
            
            # Components in prev_labeled that were not matched have ended
            # (their lifespans are already recorded)
        else:
            # First non-empty slice - components start tracking but not counted yet
            for curr_id in range(1, curr_num + 1):
                # Start tracking but don't count yet (need at least 2 slices for persistence)
                pass
        
        prev_labeled = curr_labeled
        prev_num = curr_num
    
    # Compute statistics from all tracked lifespans
    if component_lifespans:
        lifespans = list(component_lifespans.values())
        mean_persistence = float(np.mean(lifespans))
        median_persistence = float(np.median(lifespans))
        max_persistence = int(np.max(lifespans))
    else:
        # No components persisted for ≥2 consecutive slices
        mean_persistence = 0.0
        median_persistence = 0.0
        max_persistence = 0
    
    return {
        'mean_persistence': mean_persistence,
        'median_persistence': median_persistence,
        'max_persistence': max_persistence
    }
