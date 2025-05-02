import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm

# Import project modules
from config.default_config import parse_args, EDM_CONFIG
from data.dataset import FullFieldDataset
from models.unet import create_unet_model, setup_model_devices, load_vae, load_classifier, load_clip_model
from schedulers.edm_scheduler import create_edm_scheduler, prepare_input_with_noise, calculate_edm_loss
from utils.logging_utils import setup_logging, configure_diffusers_logging, log_training_parameters
from utils.checkpoint_utils import setup_checkpoint_hooks, save_checkpoint, cleanup_checkpoints, resume_from_checkpoint
from utils.edm_utils import edm_precondition, edm_loss_weight, prepare_latent_sample, prepare_model_inputs, edm_clean_image_to_model_input, edm_model_output_to_x_0_hat

# Import required diffusers components
from diffusers import DiffusionPipeline, DDPMPipeline
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_tensorboard_available, is_wandb_available


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Get EDM config values
    sigma_min = EDM_CONFIG["SIGMA_MIN"]
    sigma_max = EDM_CONFIG["SIGMA_MAX"]
    sigma_data = EDM_CONFIG["SIGMA_DATA"]
    rho = EDM_CONFIG["RHO"]
    
    # Setup logging
    logger = setup_logging(__name__)
    
    # Setup accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    # Initialize accelerator with longer timeout for high-resolution or big datasets
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    # Check if selected logger is available
    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    
    # Setup checkpoint hooks for accelerator
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        accelerator = setup_checkpoint_hooks(accelerator, args)
    
    # Configure logging
    logger.info(accelerator.state, main_process_only=False)
    configure_diffusers_logging(accelerator.is_local_main_process)
    
    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Set weight dtype based on precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Create UNet model
    model = create_unet_model(args.model_config_name_or_path, args.resolution)
    
    # Print model parameter count
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {pytorch_total_params}")
    
    # Create EMA model if needed
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=type(model),
            model_config=model.config,
        )
    
    # Setup xformers if requested
    if args.enable_xformers_memory_efficient_attention:
        try:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ImportError("xformers is not available")
        except ImportError as e:
            logger.warning(f"{e}. xformers will not be used.")
    
    # Create noise scheduler
    noise_scheduler = create_edm_scheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        num_train_timesteps=args.ddpm_num_steps,
        prediction_type=args.prediction_type
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Create dataset and dataloader
    dataset = FullFieldDataset(
        data_root='/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/training_images',
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader_num_workers
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    # Load VAE and classifier
    vae = load_vae(
        vae_path="/scratch/groups/emmalu/marvinli/twisted_diffusion/stable-diffusion-3.5-large-turbo/vae",  # Update with your VAE path
        accelerator=accelerator,
        weight_dtype=weight_dtype
    )
    
    classifier_location = load_classifier(
        checkpoint_path="/scratch/groups/emmalu/marvinli/twisted_diffusion/checkpoints_classifier/model_epoch_7.pth",  # Update with your classifier path
        accelerator=accelerator,
        weight_dtype=weight_dtype
    )
    
    clip_model = load_clip_model(
        clip_model_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        accelerator=accelerator,
        weight_dtype=weight_dtype
    )

    # Prepare components with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move EMA model to device if used
    if args.use_ema:
        ema_model.to(accelerator.device)
        accelerator.ema_model = ema_model
    
    # Move scheduler sigmas to device
    noise_scheduler.sigmas = noise_scheduler.sigmas.to(accelerator.device)
    
    # Initialize trackers
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)
    
    # Calculate training parameters
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    # Log training parameters
    log_training_parameters(
        logger, args, total_batch_size, num_update_steps_per_epoch, max_train_steps, len(dataset)
    )
    
    # Resume from checkpoint if requested
    global_step, first_epoch, resume_step = resume_from_checkpoint(accelerator, args)
    
    if args.use_ema:
        ema_model = accelerator.ema_model
    
    # Training loop
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            # Process VAE encoding
            with torch.no_grad():
                # Encode conditional images
                cond_images = batch["cond_image"].to(weight_dtype)
                cond_images_latent = prepare_latent_sample(vae, cond_images, weight_dtype)
                
                # Encode ground truth images (repeat single channel to 3 channels if needed)
                gt_images = batch["gt_image"].repeat(1, 3, 1, 1).to(weight_dtype)
                gt_images_latent = prepare_latent_sample(vae, gt_images, weight_dtype)

                clip_images = batch["clip_image"].to(weight_dtype)
                encoder_hidden_states = clip_model(clip_images)
                # Prepare model inputs (combined latents and labels)
                clean_images, total_label, encoder_hidden_states, dropout_mask = prepare_model_inputs(
                    gt_images_latent, 
                    cond_images_latent, 
                    batch["cell_line"], 
                    batch["label"],
                    dropout_prob=0.2,
                    weight_dtype=weight_dtype,
                    encoder_hidden_states=encoder_hidden_states,
                )
                
                # Get annotation ground truth
                ground_truth_location = batch["annotation"]
            
            # Sample noise and timesteps
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()
            
            # Get sigmas for the selected timesteps
            sigmas = noise_scheduler.sigmas[timesteps].reshape(-1, 1, 1, 1).to(weight_dtype)
            #P_std = 1.2
            #P_mean = -1.2
            #sigmas = ((noise * P_std + P_mean).exp()).reshape(-1, 1, 1, 1).to(weight_dtype)



            x_noisy = noise * sigmas + clean_images


            
            # Gradient accumulation training loop
            with accelerator.accumulate(model):
                # Model prediction (pass sigma directly instead of timestep indices)
                # Add noise according to EDM formulation
                model_input, timestep_input = edm_clean_image_to_model_input(x_noisy, sigmas)
                timestep_input = timestep_input.squeeze()
                
                
                # Get model output
                model_output = model(
                    model_input, 
                    timestep_input,
                    class_labels=total_label,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                x_0_hat = edm_model_output_to_x_0_hat(x_noisy, sigmas, model_output)

                # only predict the x0 is allowed
                target = clean_images

                
                # Calculate EDM loss
                weights = edm_loss_weight(sigmas)
                loss = weights*((x_0_hat.float() - target.float()) ** 2)
                loss = loss.mean()

                # Optional location loss using classifier
                if False:  # Set to True to enable location loss
                    # Generate predicted clean image
                    
                    # Reshape and pass through location classifier
                    x_0_hat = x_0_hat.reshape(-1, 32, 32, 32)
                    x_0_hat = x_0_hat * 4 / vae.scaling_factor
                    image_predicted_location = classifier_location(x_0_hat.to(weight_dtype))
                    
                    # Calculate location loss
                    loss_location = F.binary_cross_entropy_with_logits(
                        image_predicted_location, ground_truth_location, reduction="none"
                    )
                    loss_location = loss_location[dropout_mask].mean()
                    # Combine losses (adjust weight as needed)
                    loss = loss + 0.01 * loss_location
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update EMA model if used
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                
                # Save checkpoint periodically
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    # Clean up old checkpoints if needed
                    cleanup_checkpoints(args.output_dir, args.checkpoints_total_limit)
                    
                    # Save checkpoint
                    save_path = save_checkpoint(accelerator, args, global_step)
                    logger.info(f"Saved state to {save_path}")
            
            # Log metrics
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
        # End of epoch
        progress_bar.close()
        accelerator.wait_for_everyone()
        
        # Generate sample images for visualization
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)
                
                # Use EMA model if available
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                
                # Create pipeline for inference
                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )
                
                # Optional: Generate sample images for visualization
                # Implementation depends on your specific inference needs
                # This would be similar to the inference.py script
                
                # Restore original model parameters if EMA was used
                if args.use_ema:
                    ema_model.restore(unet.parameters())
            
            # Save model periodically
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # Unwrap model
                unet = accelerator.unwrap_model(model)
                
                # Use EMA model if available
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                
                # Create and save pipeline
                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )
                
                pipeline.save_pretrained(args.output_dir)
                
                # Restore original model parameters if EMA was used
                if args.use_ema:
                    ema_model.restore(unet.parameters())
                
                # Push to hub if requested
                if args.push_to_hub:
                    from huggingface_hub import upload_folder
                    
                    upload_folder(
                        repo_id=args.hub_model_id or Path(args.output_dir).name,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
    
    # End of training
    accelerator.end_training()


if __name__ == "__main__":
    main()