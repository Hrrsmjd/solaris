from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from torch.utils.data import DataLoader

from solaris.load_data import CustomDataset_downstream
from solaris.normalization import transform
from solaris.utils_data import build_metadata

def rmse(predictions, ground_truth):
    """Calculate Root Mean Square Error between predictions and ground truth."""
    return torch.sqrt(torch.mean((predictions - ground_truth)**2))

def loss(predictions, ground_truth, scale_factor):
    """Calculate scaled Mean Absolute Error loss."""
    mae_loss = torch.abs(predictions - ground_truth).mean()  
    scaled_loss = mae_loss / scale_factor
    return scaled_loss

def model_eval(model, test_dataset, norm_coeff_1, norm_coeff_2, input_scale, output_scale):
    """Evaluate model performance on downstream task dataset."""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    total_loss = 0.0
    total_rmse = 0.0
    num_samples = 0
    
    all_predictions = []
    all_truths = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = transform(data, norm_coeff_1, norm_coeff_2, input_scale)
            target = transform(target, norm_coeff_1, norm_coeff_2, output_scale)
            
            metadata = build_metadata(data)
            prediction = model(data.unsqueeze(1), metadata, 12, 0).squeeze(1)
            
            batch_loss = loss(prediction, target, output_scale)
            batch_rmse = rmse(prediction, target)
            
            total_loss += batch_loss.item()
            total_rmse += batch_rmse.item()
            num_samples += 1
            
            all_predictions.append(prediction.cpu().numpy())
            all_truths.append(target.cpu().numpy())
    
    avg_loss = total_loss / num_samples
    avg_rmse = total_rmse / num_samples
    
    predictions = np.concatenate(all_predictions, axis=0)
    truths = np.concatenate(all_truths, axis=0)
    
    ssim_scores = []
    for i in range(len(predictions)):
        pred_img = predictions[i, 0]
        truth_img = truths[i, 0]
        ssim_score = ssim(pred_img, truth_img, data_range=truth_img.max() - truth_img.min())
        ssim_scores.append(ssim_score)
    
    avg_ssim = np.mean(ssim_scores)
    
    return {
        'loss': avg_loss,
        'rmse': avg_rmse,
        'ssim': avg_ssim,
        'predictions': predictions,
        'truths': truths
    }
