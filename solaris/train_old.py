import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from solaris.load_data import CustomDataset_pretrain as CustomDataset
from solaris.model.solaris import Solaris
from solaris.utils_data import build_metadata

if __name__ == '__main__':

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
    
    train_dataset = CustomDataset(root_dir=args.data_path, data_set="train", id_dir=args.id_dir)
    val_dataset = CustomDataset(root_dir=args.data_path, data_set="val", id_dir=args.id_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = Solaris(out_levels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            metadata = build_metadata(data)
            output = model(data.unsqueeze(1), metadata, 12, 0).squeeze(1)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        scheduler.step()
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.model_save_path)
            print(f'Model saved at epoch {epoch}')
