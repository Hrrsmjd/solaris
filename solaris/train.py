import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from solaris.load_data import CustomDataset_pretrain
from solaris.model.solaris import Solaris
from solaris.normalization import transform
from solaris.utils_data import build_metadata

def percentage_of_good_pixels(pred: torch.Tensor, truth: torch.Tensor, t: float) -> float:
    """Calculate the percentage of good pixels metric."""
    t_decimal = t / 100.0
    
    diff = torch.abs(pred - truth)
    
    good_pixels = diff <= t_decimal
    
    total_pixels = torch.numel(pred)
    good_pixel_count = torch.sum(good_pixels).item()
    
    percentage = (good_pixel_count / total_pixels) * 100.0
    
    return percentage

def rmse(predictions, ground_truth):
    """Calculate Root Mean Square Error between predictions and ground truth."""
    return torch.sqrt(torch.mean((predictions - ground_truth)**2))

def loss(predictions, ground_truth, scale_factor):
    """Calculate scaled Mean Absolute Error loss."""
    mae_loss = torch.abs(predictions - ground_truth).mean()  
    scaled_loss = mae_loss / scale_factor
    return scaled_loss

def epoch_train(model, train_loader, optimizer, device, norm_coeff_1, norm_coeff_2, input_scale, output_scale):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        data = transform(data, norm_coeff_1, norm_coeff_2, input_scale)
        target = transform(target, norm_coeff_1, norm_coeff_2, output_scale)
        
        optimizer.zero_grad()
        
        metadata = build_metadata(data)
        output = model(data.unsqueeze(1), metadata, 12, 0).squeeze(1)
        
        batch_loss = loss(output, target, output_scale)
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {batch_loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def epoch_eval(model, val_loader, device, norm_coeff_1, norm_coeff_2, input_scale, output_scale):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_good_pixels = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            data = transform(data, norm_coeff_1, norm_coeff_2, input_scale)
            target = transform(target, norm_coeff_1, norm_coeff_2, output_scale)
            
            metadata = build_metadata(data)
            output = model(data.unsqueeze(1), metadata, 12, 0).squeeze(1)
            
            batch_loss = loss(output, target, output_scale)
            batch_rmse = rmse(output, target)
            batch_good_pixels = percentage_of_good_pixels(output, target, 10.0)
            
            total_loss += batch_loss.item()
            total_rmse += batch_rmse.item()
            total_good_pixels += batch_good_pixels
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_rmse = total_rmse / num_batches
    avg_good_pixels = total_good_pixels / num_batches
    
    return avg_loss, avg_rmse, avg_good_pixels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--id_dir', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default='./model_checkpoint.pth')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_dataset = CustomDataset_pretrain(root_dir=args.data_path, data_set="train", id_dir=args.id_dir)
    val_dataset = CustomDataset_pretrain(root_dir=args.data_path, data_set="val", id_dir=args.id_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = Solaris(out_levels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    norm_coeff_1 = torch.tensor(1.0)  # First normalization coefficient
    norm_coeff_2 = torch.tensor(1.0)  # Second normalization coefficient
    input_scale = torch.ones(6)       # Scaling for 6 input wavelength channels
    output_scale = torch.ones(1)      # Scaling for single output channel
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        train_loss = epoch_train(model, train_loader, optimizer, device, norm_coeff_1, norm_coeff_2, input_scale, output_scale)
        val_loss, val_rmse, val_good_pixels = epoch_eval(model, val_loader, device, norm_coeff_1, norm_coeff_2, input_scale, output_scale)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.6f}, Val Good Pixels: {val_good_pixels:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f'Model saved with validation loss: {val_loss:.6f}')

if __name__ == '__main__':
    main()
