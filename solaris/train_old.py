import argparse
import torch
import numpy as np
from solaris.model.solaris import Solaris
from solaris.normalization import transform
import random
import time
import datetime
from solaris.load_data import CustomDataset_pretrain as CustomDataset
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from solaris.optimizer import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='CHANGE_PATH')
    parser.add_argument('--model_save_path', type=str, default='./model_checkpoint.pth')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_dataset = CustomDataset(root_dir=args.data_path, data_set="train")
    val_dataset = CustomDataset(root_dir=args.data_path, data_set="val")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = Solaris().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        scheduler.step()
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.model_save_path)
            print(f'Model saved at epoch {epoch}')
