import torch
import h5py

def transform(input_data, coeff_1, coeff_2, scale_factors):
    """
    Apply normalization transform to solar data.
    
    Args:
        input_data: Input tensor to normalize
        coeff_1: First normalization coefficient
        coeff_2: Second normalization coefficient  
        scale_factors: Scaling factors for each channel
        
    Returns:
        Normalized tensor
    """
    scaled_data = input_data / scale_factors[None, None, :, None, None]
    
    epsilon = torch.tensor(1e-3)  # Small value to prevent log(0)
    clipped_data = torch.minimum(scaled_data, torch.tensor(2.5))
    log_term = (torch.log(torch.maximum(scaled_data, epsilon)) - torch.log(epsilon)) / torch.log(epsilon)
    
    normalized_data = coeff_1 * clipped_data - coeff_2 * log_term
    return normalized_data
