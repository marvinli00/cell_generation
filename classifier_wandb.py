import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from data.dataset import FullFieldDataset
import pickle
from diffusers import AutoencoderKL
from tqdm import tqdm
import os

def top_k_accuracy(output, target, k=1):
    """Fixed version - returns both accuracy and correct count"""
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
            in_chans=16  # timm supports direct specification of input channels
        )
        
        # Model expects 224x224 input, so add an adapter for 32x32
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Input shape: [batch_size, 32, 32, 32]
        x = self.resize(x)  # Resize to [batch_size, 32, 224, 224]
        x1 = self.vit(x)
        return x1

class Trainer:
    def __init__(self, model, train_loader, val_loader, config=None):
        # Initialize config with defaults if not provided
        self.config = {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_epochs': 50,
           'batch_size': 256,
           'model_type': 'vit_small_patch16_224',
           'optimizer': 'Adam',
           'scheduler': 'linear_with_warmup',
           'input_channels': 32,
           'input_size': 64,
           'wandb_project': 'location-classifier',
           'wandb_run_name': None,
           'gradient_accumulation_steps': 1,
           'mixed_precision': 'bf16',
        } if config is None else config
       
        # Initialize accelerator with wandb logging
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            mixed_precision=self.config['mixed_precision'],
            log_with="wandb",
        )
        
        # Initialize wandb through accelerator
        # This is the proper way to initialize wandb with accelerator
        self.accelerator.init_trackers(
            project_name=self.config['wandb_project'],
            config=self.config,
            init_kwargs={
                "wandb": {
                    "name": self.config.get('wandb_run_name'),
                    "tags": ["vit", "classification", "location"],
                    "notes": f"Training {self.config['model_type']} on location classification",
                }
            }
        )
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Define weight_dtype based on mixed precision
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # Load VAE
        vae = AutoencoderKL.from_pretrained("/scratch/groups/emmalu/marvinli/twisted_diffusion/stable-diffusion-3.5-large-turbo/vae")
        vae.requires_grad_(False)
        vae.eval()
        
        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Prepare everything with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        
        # Move VAE to device after prepare (to ensure correct device)
        self.vae = vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Track global step
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Create progress bar only on main process
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, data in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                target = data['label']
                batch_size = target.shape[0]

                # Process images through VAE
                # cond_images = data['cond_image'].to(self.weight_dtype)
                gt_images = data["gt_image"].repeat(1, 3, 1, 1).to(self.weight_dtype)
                
                with torch.no_grad():
                    # cond_images_latent = self.vae.encode(cond_images).latent_dist.sample()
                    gt_images_latent = self.vae.encode(gt_images).latent_dist.sample()

                # images = torch.cat([gt_images_latent, cond_images_latent], dim=1)
                
                images = gt_images_latent*self.vae.scaling_factor/4
                # Forward pass
                output = self.model(images)
                loss = self.criterion(output, target)
                
                # Backward pass
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Gather loss across all processes
                gathered_loss = self.accelerator.gather(loss.detach())
                total_loss += gathered_loss.mean().item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Log batch metrics
                if self.global_step % 10 == 0:
                    self.accelerator.log({
                        "train/batch_loss": loss.detach().item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "train/global_step": self.global_step,
                    }, step=self.global_step)
                
                self.global_step += 1
        
        # Calculate epoch average
        avg_loss = total_loss / total_samples
        
        # Log epoch metrics
        self.accelerator.log({
            "train/epoch_loss": avg_loss,
            "train/epoch": epoch,
        }, step=self.global_step)
        
        return avg_loss


    def validate(self, epoch):
        # Ensure all processes wait and sync model
        self.accelerator.wait_for_everyone()
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_correct = torch.zeros(4, device=self.accelerator.device)  # For top-1, 10, 20, 50
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Valid]",
            disable=not self.accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for data in progress_bar:
                target = data['label']
                batch_size = target.shape[0]

                # Process images through VAE
                # cond_images = data['cond_image'].to(self.weight_dtype)
                gt_images = data["gt_image"].repeat(1, 3, 1, 1).to(self.weight_dtype)
                
                # cond_images_latent = self.vae.encode(cond_images).latent_dist.sample()
                gt_images_latent = self.vae.encode(gt_images).latent_dist.sample()

                # images = torch.cat([gt_images_latent, cond_images_latent], dim=1)

                images = gt_images_latent * self.vae.scaling_factor / 4 

                # Forward pass
                output = self.model(images)
                loss = self.criterion(output, target)
                
                # Gather loss across all processes for correct averaging
                gathered_loss = self.accelerator.gather(loss.detach())
                total_loss += gathered_loss.mean().item() * batch_size
                
                # Calculate accuracies - FIXED VERSION
                for i, k in enumerate([1, 10, 20, 50]):
                    acc, correct_count = top_k_accuracy(output, target, k=k)
                    # Gather correct counts across all processes
                    gathered_correct = self.accelerator.gather(correct_count)
                    total_correct[i] += gathered_correct.sum()
                
                # Gather batch sizes across all processes
                gathered_batch_size = self.accelerator.gather(torch.tensor(batch_size, device=self.accelerator.device))
                total_samples += gathered_batch_size.sum().item()
                
                # Update progress bar (only on main process)
                if self.accelerator.is_local_main_process:
                    current_acc = total_correct[0].item() / total_samples if total_samples > 0 else 0
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples
        accuracies = (total_correct / total_samples).cpu().numpy()

        print(total_correct, total_samples, accuracies)
        
        # Only log on main process
        if self.accelerator.is_main_process:
            metrics = {
                "val/loss": avg_loss,
                "val/top1_accuracy": accuracies[0],
                "val/top10_accuracy": accuracies[1],
                "val/top20_accuracy": accuracies[2],
                "val/top50_accuracy": accuracies[3],
                "val/epoch": epoch,
            }
            
            self.accelerator.log(metrics, step=self.global_step)
        
        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        return avg_loss, accuracies


    def save_model(self, epoch, val_loss):
        # Only save on main process
        if self.accelerator.is_main_process:
            checkpoint_path = Path('checkpoints_classifier') / f'model_epoch_{epoch}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            # Unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            checkpoint = {
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': self.config,
                'global_step': self.global_step,
            }
            
            # Use accelerator.save to handle distributed saving
            self.accelerator.save(checkpoint, checkpoint_path)
            
            # Log checkpoint info
            self.accelerator.log({
                "checkpoint/epoch": epoch,
                "checkpoint/val_loss": val_loss,
            }, step=self.global_step)

    def train(self):
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate(epoch)
            
            # Log summary metrics
            self.accelerator.log({
                "summary/train_loss": train_loss,
                "summary/val_loss": val_loss,
                "summary/val_top1_accuracy": val_accuracy[0],
                "summary/epoch": epoch,
            }, step=self.global_step)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
                
                # Log best metrics
                self.accelerator.log({
                    "best/val_loss": best_val_loss,
                    "best/val_top1_accuracy": val_accuracy[0],
                    "best/epoch": epoch,
                }, step=self.global_step)
            
            # Print summary
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Top-1 Accuracy: {val_accuracy[0]:.4f}")
                print(f"Val Top-10 Accuracy: {val_accuracy[1]:.4f}")
                print(f"Val Top-20 Accuracy: {val_accuracy[2]:.4f}")
                print(f"Val Top-50 Accuracy: {val_accuracy[3]:.4f}")
                print(f"Best Val Loss: {best_val_loss:.4f}\n")
        
        # Finish tracking
        self.accelerator.end_training()

def main():
    # Set environment variables for wandb (optional but recommended)
    os.environ["WANDB_LOG_MODEL"] = "false"  # We'll handle model saving ourselves
    
    # Define configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 50,
        'batch_size': 100,
        'num_classes': 12810,
        'model_type': 'vit_small_patch16_224',
        'optimizer': 'AdamW',
        'scheduler': 'linear_with_warmup',
        'input_channels': 32,
        'input_size': 64,
        'wandb_project': 'location-classifier',
        'wandb_run_name': 'vit-location-classification',
        'gradient_accumulation_steps': 1,
        'mixed_precision': 'bf16',
    }
    
    # Initialize model
    model = LocationClassifier(
        num_classes=config['num_classes'],
        pretrained=True,
        model_type=config['model_type']
    )
    
    # Initialize datasets
    train_dataset = FullFieldDataset(data_root='/scratch/groups/emmalu/multimodal_phenotyping/dataset/images/',
                                    label_dict = "/scratch/groups/emmalu/multimodal_phenotyping/cell_line_map.pkl",
                                    annotation_dict = "/scratch/groups/emmalu/multimodal_phenotyping/antibody_map.pkl",is_train=True)
    test_dataset = FullFieldDataset(data_root='/scratch/groups/emmalu/multimodal_phenotyping/dataset/images/',
                                    label_dict = "/scratch/groups/emmalu/multimodal_phenotyping/cell_line_map.pkl",
                                    annotation_dict = "/scratch/groups/emmalu/multimodal_phenotyping/antibody_map.pkl",is_train=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=6,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=6,
        pin_memory=False
    )
    
    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()