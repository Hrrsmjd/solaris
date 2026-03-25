from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from torch.utils.data import DataLoader

from solaris.load_data_prov import CustomDataset_pretrain
from solaris.normalization import transform
from solaris.utils_data import build_metadata

def rmse(predictions, ground_truth):
    """Calculate Root Mean Square Error using NumPy."""
    return np.sqrt(np.mean((predictions - ground_truth)**2))

def mean_absolute_error(predictions, ground_truth):
    """Calculate Mean Absolute Error using PyTorch."""
    return torch.abs(ground_truth - predictions).mean(dim=0)

def percentage_of_good_pixels(pred: torch.Tensor, truth: torch.Tensor, t: float) -> float:
    """Calculate the percentage of good pixels metric."""
    t_decimal = t / 100.0
    
    diff = torch.abs(pred - truth)
    
    good_pixels = diff <= t_decimal
    
    total_pixels = torch.numel(pred)
    good_pixel_count = torch.sum(good_pixels).item()
    
    percentage = (good_pixel_count / total_pixels) * 100.0
    
    return percentage

def model_eval(model, test_dataset, norm_coeff_1, norm_coeff_2, input_scale, output_scale):
    """Evaluate model performance on test dataset with comprehensive metrics."""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_predictions = []
    all_truths = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = transform(data, norm_coeff_1, norm_coeff_2, input_scale)
            target = transform(target, norm_coeff_1, norm_coeff_2, output_scale)
            
            metadata = build_metadata(data)
            prediction = model(data.unsqueeze(1), metadata, 12, 0).squeeze(1)
            
            all_predictions.append(prediction.cpu().numpy())
            all_truths.append(target.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    truths = np.concatenate(all_truths, axis=0)
    
    rmse_score = rmse(predictions, truths)
    mae_score = np.mean(np.abs(predictions - truths))
    
    good_pixels_10 = percentage_of_good_pixels(torch.from_numpy(predictions), torch.from_numpy(truths), 10.0)
    good_pixels_20 = percentage_of_good_pixels(torch.from_numpy(predictions), torch.from_numpy(truths), 20.0)
    
    ssim_scores = []
    for i in range(len(predictions)):
        pred_img = predictions[i, 0]
        truth_img = truths[i, 0]
        ssim_score = ssim(pred_img, truth_img, data_range=truth_img.max() - truth_img.min())
        ssim_scores.append(ssim_score)
    
    avg_ssim = np.mean(ssim_scores)
    
    return {
        'rmse': rmse_score,
        'mae': mae_score,
        'good_pixels_10': good_pixels_10,
        'good_pixels_20': good_pixels_20,
        'ssim': avg_ssim
    }

def save_sample(model, test_dataset, save_path, norm_coeff_1, norm_coeff_2, input_scale, output_scale, num_samples=5):
    """Save model predictions and ground truth for a specified number of samples."""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    samples_saved = 0
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if samples_saved >= num_samples:
                break
                
            data = transform(data, norm_coeff_1, norm_coeff_2, input_scale)
            target = transform(target, norm_coeff_1, norm_coeff_2, output_scale)
            
            metadata = build_metadata(data)
            prediction = model(data.unsqueeze(1), metadata, 12, 0).squeeze(1)
            
            sample_data = {
                'input': data.cpu().numpy(),
                'prediction': prediction.cpu().numpy(),
                'truth': target.cpu().numpy()
            }
            
            np.save(f"{save_path}/sample_{idx}.npy", sample_data)
            samples_saved += 1
    
    print(f"Saved {samples_saved} samples to {save_path}")
