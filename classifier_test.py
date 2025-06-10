# CRITICAL BUGS IDENTIFIED IN YOUR TEST CODE:

# BUG 1: Double counting total_samples
# You increment total_samples twice in the loop:
# total_samples += batch_size  # First time
# ... some code ...  
# total_samples += batch_size  # Second time - WRONG!

# BUG 2: Wrong top-1 accuracy calculation
# correct[0] += acc * batch_size  # acc is already a count, not ratio!
# This inflates your accuracy by batch_size

# BUG 3: Loss calculation is wrong
# loss = F.cross_entropy(output, target).mean() * batch_size
# cross_entropy already returns mean loss, you're multiplying by batch_size incorrectly

# BUG 4: Inconsistent accuracy calculation methods
# Top-1 uses count * batch_size, top-k uses ratio * batch_size

# FIXED VERSION:

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from data.dataset import FullFieldDataset
from diffusers import AutoencoderKL
from tqdm import tqdm
import os

def top_k_accuracy(output, target, k=1):
    """Returns accuracy ratio and correct count"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracy = correct_k / batch_size
        return accuracy, correct_k

class LocationClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_type='vit_small_patch16_224'):
        super().__init__()
        
        self.model_type = model_type
        
        # Load pretrained ViT from timm
        self.vit = timm.create_model(
            model_type, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=32  # timm supports direct specification of input channels
        )
        
        # Model expects 224x224 input, so add an adapter for 32x32
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Input shape: [batch_size, 32, 32, 32]
        x = self.resize(x)  # Resize to [batch_size, 32, 224, 224]
        x1 = self.vit(x)
        return x1

def test_model():
    """Fixed test function"""
    # Define configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 50,
        'batch_size': 64,
        'num_classes': 13348,
        'model_type': 'vit_small_patch16_224',
        'optimizer': 'AdamW',
        'scheduler': 'linear_with_warmup',
        'input_channels': 32,
        'input_size': 32,
        'wandb_project': 'location-classifier',
        'wandb_run_name': 'vit-location-classification',
        'gradient_accumulation_steps': 1,
        'mixed_precision': 'bf16',
    }
    checkpoint_path = "model_epoch_21.pth"
    # Initialize model
    model = LocationClassifier(
        num_classes=config['num_classes'],
        pretrained=True,
        model_type=config['model_type']
    )
    # Load pretrained weights if available

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint['config']
    model.load_state_dict(checkpoint['model_state_dict'])

    model.cuda()
    model.eval()
    
    # Initialize dataset
    test_dataset = FullFieldDataset(data_root='./test_images_2')
    
    # Initialize dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("../stable-diffusion-3.5-large-turbo/vae")
    vae.requires_grad_(False)
    vae.eval()
    vae.cuda()
    
    # FIXED TEST LOOP
    with torch.no_grad():
        model.eval()
        total_correct = torch.zeros(4, device='cuda')  # For top-1, 10, 20, 50
        total_samples = 0
        total_loss = 0.0
        
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing")):
            target = data['label'].cuda()
            batch_size = target.shape[0]

            # Process images through VAE
            cond_images = data['cond_image'].cuda()
            gt_images = data["gt_image"].repeat(1, 3, 1, 1).cuda()

            # print filenames for debugging
            print(data["path"])

            cond_images_latent = vae.encode(cond_images).latent_dist.sample()
            gt_images_latent = vae.encode(gt_images).latent_dist.sample()

            images = torch.cat([gt_images_latent, cond_images_latent], dim=1)

            # Forward pass  
            output = model(images)

            # FIXED: Calculate loss correctly
            batch_loss = F.cross_entropy(output, target)  # Already mean loss
            total_loss += batch_loss.item() * batch_size  # Accumulate total loss
            
            # FIXED: Calculate top-1 accuracy consistently
            _, pred_top1 = output.topk(1, 1, True, True)
            pred_top1 = pred_top1.squeeze(1)
            correct_top1 = (pred_top1 == target).float().sum()
            total_correct[0] += correct_top1  # Add count, not ratio

            print("target:", target)
            print("pred_top1:", pred_top1)
            
            # FIXED: Calculate top-k accuracies consistently
            for i, k in enumerate([10, 20, 50]):
                _, correct_k = top_k_accuracy(output, target, k=k)
                total_correct[i + 1] += correct_k  # Add count, not ratio
            
            # FIXED: Increment total_samples only once
            total_samples += batch_size
            
            # Debug print for first few batches
            if batch_idx < 3:
                batch_acc = correct_top1.item() / batch_size
                print(f"Batch {batch_idx}: Loss={batch_loss.item():.4f}, "
                      f"Top-1 Acc={batch_acc:.4f}, "
                      f"Correct={correct_top1.item()}/{batch_size}")
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples
        accuracies = (total_correct / total_samples).cpu().numpy()
        
        print(f"\n=== FINAL TEST RESULTS ===")
        print(f"Total samples: {total_samples}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Test Top-1 Accuracy: {accuracies[0]:.4f}")
        print(f"Test Top-10 Accuracy: {accuracies[1]:.4f}")
        print(f"Test Top-20 Accuracy: {accuracies[2]:.4f}")
        print(f"Test Top-50 Accuracy: {accuracies[3]:.4f}")
        
        # Additional debugging info
        print(f"\n=== DEBUGGING INFO ===")
        print(f"Total correct (top-1): {total_correct[0].item()}")
        print(f"Expected accuracy range for 13348 classes: ~0.0001 (random)")
        
        # Sanity check
        if accuracies[0] > 0.1:  # If accuracy > 10%, something might be wrong
            print("WARNING: Accuracy seems suspiciously high for 13348 classes!")
        elif accuracies[0] < 1e-6:  # If accuracy < 0.0001%, something might be wrong
            print("WARNING: Accuracy seems suspiciously low!")
        else:
            print("Accuracy seems reasonable for the number of classes.")

def debug_data_and_model():
    """Additional debugging function"""
    test_dataset = FullFieldDataset(data_root='./test_images_2')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Check data
    sample_batch = next(iter(test_loader))
    print(f"=== DATA DEBUG ===")
    print(f"Batch size: {len(sample_batch['label'])}")
    print(f"Label range: [{sample_batch['label'].min()}, {sample_batch['label'].max()}]")
    print(f"Unique labels in batch: {len(torch.unique(sample_batch['label']))}")
    print(f"cond_image shape: {sample_batch['cond_image'].shape}")
    print(f"gt_image shape: {sample_batch['gt_image'].shape}")
    
    # Check if labels are valid
    max_label = sample_batch['label'].max().item()
    if max_label >= 13348:
        print(f"ERROR: Label {max_label} exceeds num_classes-1 ({13348-1})")

if __name__ == "__main__":
    print("Running data and model debug...")
    debug_data_and_model()
    
    print("\nRunning fixed test...")
    test_model()