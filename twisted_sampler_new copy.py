import os
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
from tifffile import imwrite

import pandas as pd

from functools import partial
import wandb
import tqdm

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path


# Import project modules
from models.unet import create_unet_model, load_vae, load_classifier, load_clip_model, CustomUNetWithEmbeddings
from schedulers.edm_scheduler import create_edm_scheduler
from utils.edm_utils import edm_clean_image_to_model_input, edm_model_output_to_x_0_hat
from config.default_config import EDM_CONFIG
from models.clip_image_encoder import OpenCLIPVisionEncoder
from data.dataset import FullFieldDataset
from models.dit import create_dit_model, DiTTransformer2DModelWithCrossAttention
from utils.edm_utils import edm_clean_image_to_model_input, edm_model_output_to_x_0_hat





def compute_ess(w, dim=0):
    ess = (w.sum(dim=dim))**2 / torch.sum(w**2, dim=dim)
    return ess

def compute_ess_from_log_w(log_w, dim=0):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)

def normalize_weights(log_weights, dim=0):
    return torch.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights

def normalize_log_weights_everything(log_weights_list, dim):
    return [normalize_log_weights(log_weights, dim) for log_weights in log_weights_list]

def log_normal_density(sample, mean, var):
    return Normal(loc=mean, scale=torch.sqrt(var)).log_prob(sample)

def systematic_resampling(particles, weights):
    """
    Perform systematic resampling on a set of particles based on their weights.
    
    Args:
        particles (numpy.ndarray): Array of particles to be resampled (N x D)
        weights (numpy.ndarray): Array of particle weights (N,)
        
    Returns:
        tuple:
            - numpy.ndarray: Resampled particles
            - numpy.ndarray: New uniform weights
            - numpy.ndarray: Indices of selected particles
    """
    N = len(weights)
    # Normalize weights
    weights /= torch.sum(weights)
    
    # Calculate cumulative sum of weights
    cumsum = torch.cumsum(weights, dim = 0)
    
    # Generate systematic noise (one random number)
    u = torch.distributions.Uniform(low=0.0, high=1.0/N).sample()
    #u = np.random.uniform(0, 1/N)
    
    # Generate points for systematic sampling
    points = torch.zeros(N)
    for i in range(N):
        points[i] = u + i/N
    
    # Initialize arrays for results
    indexes = torch.zeros(N, dtype=int)
    cumsum = torch.cat([torch.tensor([0.0], device = device), cumsum])  # Add 0 at the beginning for easier indexing
    
    # Perform systematic resampling
    i, j = 0, 0
    while i < N:
        while points[i] > cumsum[j+1]:
            j += 1
        indexes[i] = j
        i += 1
    
    # Resample particles and reset weights
    resampled_particles = particles[indexes]
    #new_weights = torch.ones(N, device = self.device) / N
    #log new_weights
    new_weights = torch.zeros(N, device = device)
    return resampled_particles, new_weights, indexes   

def get_xstart_var(alphas_cumprod_t, tausq_=0.05,var_type = 6):
    sigmasq_ = (1-alphas_cumprod_t) / alphas_cumprod_t
    if var_type == 1:
        return sigmasq_ 
    elif var_type == 2: # pseudoinverse-guided paper https://openreview.net/forum?id=9_gsMA8MRKQ 
        tausq_ = 1.0 
        return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
        #return (1 - alphas_cumprod_t) 
    elif var_type == 5: 
        tausq_ = 0.30 
        return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
    elif var_type == 3: # DPS paper https://arxiv.org/abs/2209.14687 
        return None  
    elif var_type == 4: # pseudoinverse-guided paper -- the actual implementation, see their Alg.1 
        return beta_t  / np.sqrt(alphas_cumprod_t) 
    elif var_type == 6: # freely specify tausq_
        return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
    
def compute_ess_softmax(log_weights):
    #softmax
    weights = torch.nn.functional.softmax(log_weights, dim = 0)
    return 1/torch.sum(weights**2)

def prepare_latent_sample(vae, images, weight_dtype=torch.float32):
    """Encode images to latent space using VAE"""
    with torch.no_grad():
        latent = vae.encode(images).latent_dist.sample()
    return latent

def prepare_model_inputs(gt_images_latent, cond_images_latent, cell_line, label, dropout_prob=0.0, weight_dtype=torch.float32, encoder_hidden_states=None):
    """Prepare model inputs including latents and conditioning"""
    # Combine protein and cell line for conditioning
    batch_size = cond_images_latent.shape[0]
    
    # Create dropout mask for classifier-free guidance
    dropout_mask = torch.rand(batch_size) > dropout_prob
    
    # Create full label tensor including cell line and label
    total_label = torch.cat([cell_line, label], dim=1).to(weight_dtype)
    
    # Create a clean latent by combining ground truth and conditioning latents
    clean_images = torch.cat([gt_images_latent, cond_images_latent], dim=1)
    
    return clean_images, total_label, encoder_hidden_states, dropout_mask

def decode_latents(vae, latents, scaling_factor=4.0):
    """Decode latent samples to images using VAE"""
    with torch.no_grad():
        # Scale latents
        latents = latents * 4 / vae.scaling_factor
        
        # Decode the latents to images
        images = vae.decode(latents).sample
        
        # Normalize images to [0, 1] range
        images = (images / 2 + 0.5).clamp(0, 1)
        
    return images

def prepare_conditioning(clip_image=None, cell_line=None, label=None, batch_size=1, device="cuda", weight_dtype=torch.float32):
    """Prepare conditioning inputs"""

    num_cell_lines = 40  # Assuming 40 cell lines
    num_labels = 13348  # Assuming 13348 labels
    # Process CLIP image if provided
    if clip_image is not None:
        with torch.no_grad():
            encoder_hidden_states = clip_model(clip_image)
    else:
        # Create empty encoder hidden states
        encoder_hidden_states = torch.zeros((batch_size, 196, 768), device=device, dtype=weight_dtype)
    
    # Set up cell line and label conditioning
    if cell_line is None:
        # Create a one-hot vector for cell line 
        cell_line = torch.zeros((batch_size, num_cell_lines), device=device, dtype=weight_dtype)
        cell_line[:, 0] = 1.0  # Set first cell line as default
    
    if label is None:
        # Create a one-hot vector for label 
        label = torch.zeros((batch_size, num_labels), device=device, dtype=weight_dtype)
        label[:, 0] = 1.0  # Set first label as default
    
    total_label = torch.cat([cell_line, label], dim=1)
    
    return encoder_hidden_states, total_label
    
    
def plot_images(images, row_title=None, **kwargs):
    """Plot a grid of images and save to file without displaying"""
    if not isinstance(images, list):
        images = [images]
    
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(16, 10 // num_images))
    
    if num_images == 1:
        axs = [axs]
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # Handle different shapes and channel configurations
        if img.ndim == 4 and img.shape[0] == 1:  # [1, C, H, W]
            img = img[0]
        
        if img.shape[0] == 3 or img.shape[0] == 1:  # [C, H, W]
            img = img.transpose(1, 2, 0)
        
        if img.shape[-1] == 1:  # Single channel
            img = img.squeeze(-1)
            axs[i].imshow(img, cmap='gray')
        else:  # RGB
            axs[i].imshow(img)
        
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    if row_title is not None:
        fig.suptitle(row_title)
    
    # Save the figure with appropriate filename
    filename = f"{row_title}.png" if row_title else "plot.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to prevent display and free memory

def get_log_probs(logits, true_labels, pred_labels=None):
    """
    Get log probabilities for predicted and true labels.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        true_labels: True class indices [batch_size]
        pred_labels: Predicted class indices [batch_size]. If None, uses argmax of logits.
    
    Returns:
        dict with 'true_log_probs', 'pred_log_probs', 'all_log_probs'
    """
    # Convert logits to log probabilities
    log_probs = F.log_softmax(logits, dim=1)
    
    # Get predicted labels if not provided
    if pred_labels is None:
        pred_labels = torch.argmax(logits, dim=1)
    
    # Get log probabilities for true labels
    true_log_probs = log_probs.gather(1, true_labels.unsqueeze(1)).squeeze(1)
    
    # Get log probabilities for predicted labels
    pred_log_probs = log_probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)
    #find cross entropy loss
    #cross_entropy_loss = F.cross_entropy(logits, true_labels, reduction='none')
    return {
        'true_log_probs': true_log_probs,
        'pred_log_probs': pred_log_probs,
        'all_log_probs': log_probs
    }



def twisting_classifier(x_0_hat_scaled_to_vae, y_true, reference_channels_for_classifier, classifier, sigma_dt, number_of_particles):
    normalized_variance = get_xstart_var(sigma_dt, var_type = 6, tausq_ = 0.012)
    
    classifier_input = torch.cat([x_0_hat_scaled_to_vae.to(weight_dtype), reference_channels_for_classifier], dim=1)
    x_logit = classifier(classifier_input)
    
    return get_log_probs(x_logit, y_true)["true_log_probs"]

    #Gaussian log probability: -||x-μ||²/(2σ²)
    #score_i = -torch.sum((x_0_hat[:,:16,:,:] - mask[None,None,None,:]) ** 2, dim=(2,3)) / (2*normalized_variance)

    #score = score + score_i
    #score_log_proob_given_motif = torch.logsumexp(score, dim=0) - torch.log(torch.tensor(number_of_particles, device=device))
    
    
    #return score_log_proob_given_motif.unsqueeze(0)


def sample_edm(
    model,
    scheduler,
    batch_size=1,
    twisting_target=None,  # Target for twisting
    image_size=32,
    number_of_particles=128,
    effective_batch_size=1,  # NEW: Number of independent samples
    num_inference_steps=50,
    condition_latent=None,  # Optional conditioning latent
    encoder_hidden_states=None,  # CLIP hidden states
    protein_labels=None,
    cell_line_labels=None,
    guidance_scale=1.0,  # Scale for classifier-free guidance
    generator=None,
    output_type="latent",  # "latent" or "pt"
    
    # Add new parameters from EDMEulerScheduler.step method
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    device = None,
    weight_dtype = None,
    classifier = None,
    vae = None,
    cell_subcelluar_location = None,
    reference_channels_for_classifier = None
):      
    # Calculate total batch size: each sample gets number_of_particles
    total_batch_size = number_of_particles * effective_batch_size
    
    # Expand all conditioning inputs to match total batch size
    # Each sample's conditioning is repeated number_of_particles times
    protein_labels_expanded = []
    cell_line_labels_expanded = []
    encoder_hidden_states_expanded = []
    
    for i in range(effective_batch_size):
        # Repeat each sample's labels for all its particles
        protein_labels_expanded.append(protein_labels[i:i+1].repeat(number_of_particles))
        cell_line_labels_expanded.append(cell_line_labels[i:i+1].repeat(number_of_particles))
        encoder_hidden_states_expanded.append(encoder_hidden_states[i:i+1].repeat(number_of_particles, 1, 1))
    
    protein_labels = torch.cat(protein_labels_expanded, dim=0)
    cell_line_labels = torch.cat(cell_line_labels_expanded, dim=0)
    encoder_hidden_states = torch.cat(encoder_hidden_states_expanded, dim=0)
    
    if classifier is not None:
        cell_subcelluar_location_expanded = []
        reference_channels_expanded = []
        
        for i in range(effective_batch_size):
            cell_subcelluar_location_expanded.append(
                cell_subcelluar_location[i:i+1].repeat(number_of_particles, 1).to(weight_dtype)
            )
            reference_channels_expanded.append(
                reference_channels_for_classifier[i:i+1].repeat(number_of_particles, 1, 1, 1).to(weight_dtype)
            )
        
        cell_subcelluar_location = torch.cat(cell_subcelluar_location_expanded, dim=0)
        reference_channels_for_classifier = torch.cat(reference_channels_expanded, dim=0)
    
    # Create random noise for the ground truth part
    latent_channels = 16
    gt_noise = torch.randn(
        (total_batch_size, latent_channels, image_size, image_size),
        generator=generator,
        device=device,
        dtype=weight_dtype
        )
    latents = gt_noise * scheduler.sigmas[0].to(device)
    
    progress_bar = tqdm(range(num_inference_steps))
    progress_bar.set_description("Sampling")
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    
    #log 
    log_proposal = log_normal_density(latents, torch.tensor(0, device = device), scheduler.sigmas[0].item()**2*torch.tensor(1, device = device))
    #sum over all dim except batch and particle
    log_proposal = log_proposal.sum(dim = (1,2,3))
    
    log_proposal_tracker = []
    log_proposal_tracker.append(log_proposal)
    
    #for ess, tracing weights
    ess_tracker = []

    #initialize weights - separate for each sample
    log_w_prev_accumulated = torch.log(torch.ones_like(log_proposal, device = device))

    # Iterate
    for i, t in enumerate(progress_bar):

        sigma = scheduler.sigmas[i]
        sigma_next = scheduler.sigmas[i + 1] if i < len(scheduler.sigmas) - 1 else torch.tensor(0.0, device=device)
        
        # Expand sigma for broadcasting
        sigma_expanded = sigma.expand(total_batch_size).to(device)
        sigma_view = sigma_expanded.view(-1, 1, 1, 1).double()
        
        # Calculate gamma for stochastic sampling (from the step method)
        gamma = min(s_churn / (len(scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        sigma_hat = sigma * (gamma + 1)
        sigma_hat_view = sigma_hat.view(-1, 1, 1, 1).double().to(device)

        
        # Add noise if gamma > 0 (s_churn is active) - implements stochastic sampling
        if gamma > 0:
            noise = torch.randn((total_batch_size, latent_channels, image_size, image_size), generator=generator, device=device, dtype=latents.dtype)
            eps = noise * s_noise
            latents = latents + eps * (sigma_hat**2 - sigma**2) ** 0.5
        
        latents = latents.double()
        latents = latents.detach()
        latents.requires_grad = True
        # Combine latents with condition latent
        combined_latent = latents
        wandb.log({"latents": latents.max()}, step=i)
        # Prepare input with noise according to EDM formulation
        model_input, timestep_input = edm_clean_image_to_model_input(combined_latent, sigma_hat_view)
        timestep_input = timestep_input.squeeze()
        
        # For classifier-free guidance, we need to do two forward passes:
        # one with the conditioning and one without

        # Regular conditional forward pass
        model_input = model_input.to(weight_dtype)
        timestep_input = timestep_input.to(weight_dtype)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
        
        if len(timestep_input.shape) == 0:
            timestep_input = timestep_input.reshape(-1)
            timestep_input = timestep_input.repeat(total_batch_size)

        model_output = model(
            model_input,
            timestep_input,
            protein_labels=protein_labels,
            cell_line_labels=cell_line_labels,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
    
        # Convert model output to denoised latent (x0 prediction)
        #find E[x_0|x_t, t] unconditional
        untwisted_predicted_x_start = edm_model_output_to_x_0_hat(combined_latent, sigma_hat_view, model_output.double())

        step_sigma = sigma - sigma_next
        step_sigma = step_sigma.to(device)
        
        if classifier is not None:
            untwisted_predicted_x_start = untwisted_predicted_x_start.to(weight_dtype)
            x_0_hat_scaled_to_vae = untwisted_predicted_x_start*4/vae.scaling_factor
            log_prob_classifier = twisting_classifier(x_0_hat_scaled_to_vae, protein_labels, reference_channels_for_classifier, classifier, step_sigma, number_of_particles)
            
            # Log metrics for each sample separately
            for sample_idx in range(effective_batch_size):
                start_idx = sample_idx * number_of_particles
                end_idx = start_idx + number_of_particles
                sample_log_prob = log_prob_classifier[start_idx:end_idx]
                most_likely_index = torch.argmax(sample_log_prob, dim=0)
                wandb.log({f"log_prob_classifier_sample_{sample_idx}": sample_log_prob.mean()}, step=i)
                wandb.log({f"most_likely_index_sample_{sample_idx}": most_likely_index}, step=i)
            
        log_prob_classifier = log_prob_classifier.squeeze()
        log_prob = log_prob_classifier
        
        grad_pk_with_respect_to_x_t = torch.autograd.grad(log_prob.mean(), combined_latent)[0]
        #rescale mean back to the original scale
        grad_pk_with_respect_to_x_t = grad_pk_with_respect_to_x_t*combined_latent.shape[0]

        twisted_predicted_x_start = untwisted_predicted_x_start + grad_pk_with_respect_to_x_t
        
        twisted_predicted_x_start = twisted_predicted_x_start.detach()
        untwisted_predicted_x_start = untwisted_predicted_x_start.detach()
        twisted_predicted_x_start.requires_grad = False
        untwisted_predicted_x_start.requires_grad = False
        denoised_twisted= twisted_predicted_x_start
        denoised_untwisted = untwisted_predicted_x_start
        step_size = step_sigma / sigma
        
        direction_twisted = (denoised_twisted - latents) / sigma_view
        latents_twisted = latents + step_size.item() * sigma_view * direction_twisted
        
        with torch.no_grad():
            direction_untwisted = (denoised_untwisted - latents) / sigma_view
            latents_untwisted = latents + step_size.item() * sigma_view * direction_untwisted
        

        with torch.no_grad():
            #get p~^(t+1)_k
            log_proposal = log_proposal_tracker.pop().squeeze()
            log_proposal_tracker.append(log_prob)
            #get p~^(t)_k
            log_potential_xt = log_prob
        
            # Find p(xt_k|xt+1_k) - the reverse transition probability
            log_reverse_transition = log_normal_density(latents, latents_untwisted, step_sigma.pow(2)).sum(dim = (1,2,3))

            # Find p~(xt_k|xt+1_k,y) - the twisted reverse transition
            log_twisted_transition = log_normal_density(latents, latents_twisted, step_sigma.pow(2)).sum(dim = (1,2,3))
            temp = log_reverse_transition - log_twisted_transition
            # Calculate importance weight
            log_target = log_reverse_transition + log_potential_xt - log_twisted_transition  
            #unnormalize log_w
            log_w = log_target - log_proposal
            log_w_accumulated = log_w + log_w_prev_accumulated
            
            # Process resampling separately for each sample
            resampled_latents = []
            resampled_log_w = []
            resampled_log_proposal = []
            
            for sample_idx in range(effective_batch_size):
                start_idx = sample_idx * number_of_particles
                end_idx = start_idx + number_of_particles
                
                # Extract data for this sample
                sample_log_w_accumulated = log_w_accumulated[start_idx:end_idx]
                sample_latents_twisted = latents_twisted[start_idx:end_idx]
                sample_log_proposal = log_proposal_tracker[0][start_idx:end_idx]
                
                # Calculate ESS for this sample
                ess = compute_ess_from_log_w(sample_log_w_accumulated)
                wandb.log({f"ess_sample_{sample_idx}": ess}, step=i)
                ess_tracker.append(ess.detach().cpu().numpy())
                
                # Resample if ESS is too low
                if ess < 0.5 * number_of_particles:
                    weights = torch.nn.functional.softmax(sample_log_w_accumulated, dim=0)
                    # Resample this sample's particles
                    resampled_sample_latents, resampled_sample_log_w, indexes = systematic_resampling(
                        sample_latents_twisted, weights
                    )
                    resampled_sample_log_proposal = sample_log_proposal[indexes]
                else:
                    # No resampling needed
                    resampled_sample_latents = sample_latents_twisted
                    resampled_sample_log_w = normalize_log_weights(
                        sample_log_w_accumulated, dim=0
                    ) + torch.log(torch.tensor(number_of_particles, device=device))
                    resampled_sample_log_proposal = sample_log_proposal
                
                resampled_latents.append(resampled_sample_latents)
                resampled_log_w.append(resampled_sample_log_w)
                resampled_log_proposal.append(resampled_sample_log_proposal)
            
            # Concatenate all samples back together
            latents_twisted = torch.cat(resampled_latents, dim=0)
            log_w_prev_accumulated = torch.cat(resampled_log_w, dim=0)
            log_proposal_tracker[0] = torch.cat(resampled_log_proposal, dim=0)
            
            latents = latents_twisted
    
    # Return results with proper indexing for multiple samples
    # Find most likely particle for each sample
    most_likely_indices = []
    for sample_idx in range(effective_batch_size):
        start_idx = sample_idx * number_of_particles
        end_idx = start_idx + number_of_particles
        sample_log_prob = log_prob_classifier[start_idx:end_idx]
        most_likely_index = torch.argmax(sample_log_prob, dim=0)
        most_likely_indices.append(most_likely_index)
    
    # Reshape latents to separate samples and particles
    latents_reshaped = latents.view(effective_batch_size, number_of_particles, latent_channels, image_size, image_size)
    
    return latents_reshaped, most_likely_indices

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set precision (you can adjust this based on your hardware)
    # weight_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    weight_dtype = torch.float32  # Use float32 for compatibility with all operations

    # Define paths to models and checkpoints
    model_path = "/scratch/groups/emmalu/marvinli/twisted_diffusion/sunh/two_labels_latent_diffusion_edm_silu_cross_attention/checkpoint-640500" 
    vae_path = "/scratch/groups/emmalu/marvinli/twisted_diffusion/stable-diffusion-3.5-large-turbo/vae"
    classifier_path = "/scratch/groups/emmalu/marvinli/twisted_diffusion/checkpoints_classifier/model_epoch_21.pth"
    clip_model_path = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    # Set EDM parameters
    sigma_min = EDM_CONFIG["SIGMA_MIN"]
    sigma_max = EDM_CONFIG["SIGMA_MAX"]
    sigma_data = EDM_CONFIG["SIGMA_DATA"]
    rho = EDM_CONFIG["RHO"]

    # Set sampling parameters
    num_inference_steps = 500
    guidance_scale = 0  # Higher values increase adherence to the conditioning
    batch_size = 1
    image_size = 1  # Size of the generated images
    # Set the effective batch size
    effective_batch_size = 1  # Process MULTIPLE samples simultaneously
    number_of_particles = 32  # 16 particles per sample

    # Create model
    model = DiTTransformer2DModelWithCrossAttention.from_pretrained(model_path, subfolder="unet")
    model.to(device)
    model.to(weight_dtype)
    model.eval()
    
    # Load VAE
    class DummyAccelerator:
        def __init__(self, device):
            self.device = device

    dummy_accelerator = DummyAccelerator(device)
    vae = load_vae(vae_path, dummy_accelerator, weight_dtype)

    # Load scheduler
    scheduler = create_edm_scheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        num_train_timesteps=1000,
        prediction_type="sample"
    )

    # Move scheduler sigmas to device
    scheduler.sigmas = scheduler.sigmas.to(device)

    # Load CLIP model (optional)
    clip_model = load_clip_model(clip_model_path, dummy_accelerator, weight_dtype)

    # Load classifier (optional)
    classifier = load_classifier(classifier_path, dummy_accelerator, weight_dtype)

    print("Models loaded successfully")


    test_dataset = FullFieldDataset(
        data_root='/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/test_images',
        label_dict='/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/antibody_map.pkl',
        annotation_dict='/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/annotation_map.pkl'
    )

    # Get multiple batches of test data for effective_batch_size
    def get_batch_data(dataloader, effective_batch_size):
        """Helper function to collect multiple samples for batch processing"""
        batches = []
        dataloader_iter = iter(dataloader)
        
        for _ in range(effective_batch_size):
            try:
                batch = next(dataloader_iter)
                batches.append(batch)
            except StopIteration:
                # If we run out of data, cycle back to the beginning
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
                batches.append(batch)
        
        # Combine batches
        combined_batch = {}
        for key in batches[0].keys():
            if isinstance(batches[0][key], torch.Tensor):
                combined_batch[key] = torch.cat([b[key] for b in batches], dim=0)
            elif isinstance(batches[0][key], list):
                combined_batch[key] = [item for b in batches for item in b[key]]
            else:
                combined_batch[key] = [b[key] for b in batches]
        
        return combined_batch


    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=True,
    )

    # Get a batch of test data
    from tqdm import tqdm
    clip_model.to(weight_dtype)
    count = 0

    protein_name_list = []
    cell_line_name_list = []
    
    for batch_group in tqdm(range(0, len(test_dataset), effective_batch_size)):
        # Get batch data for multiple samples
        batch = get_batch_data(test_dataloader, effective_batch_size)
        
        # Show conditioning image
        cond_images = batch["cond_image"].to(weight_dtype).to(device)
        clip_images = batch["clip_image"].to(weight_dtype).to(device)
        gt_images = batch["gt_image"].to(weight_dtype).to(device)
        
        # Encode conditioning image to latent space
        with torch.no_grad():
            encoder_hidden_states = clip_model(clip_images)
            
        # Prepare cell_line and label conditioning
        cell_line = batch["cell_line"].to(device).long()
        protein_label = batch["label"].to(device).long()
        
        protein_names = batch["protein_name"]
        cell_line_names = batch["cell_line_name"]
        
        # Initialize wandb for this batch
        wandb.init(project="Twisted Diffusion", name=f"Twisted Diffusion Batch {batch_group}")
        wandb.log({"Protein labels": protein_names, "cell labels": cell_line_names})
        
        # Add to tracking lists
        protein_name_list.extend(protein_names)
        cell_line_name_list.extend(cell_line_names)
        
        cell_subcelluar_location = batch["annotation"].to(device).long()

        # Set random seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(42)
        twisting_target = cond_images
        
        twisting_target = prepare_latent_sample(vae, twisting_target, weight_dtype)*vae.scaling_factor/4
        reference_channels_for_classifier = vae.encode(cond_images).latent_dist.sample()
        
        # Sample from the model
        generated_latents, most_likely_indices = sample_edm(
            model=model,
            scheduler=scheduler,
            batch_size=1,
            number_of_particles=number_of_particles,
            effective_batch_size=effective_batch_size,  # NEW parameter
            twisting_target=twisting_target,
            image_size=32,
            num_inference_steps=num_inference_steps,
            condition_latent=None,
            encoder_hidden_states=encoder_hidden_states,
            protein_labels=protein_label,
            cell_line_labels=cell_line,
            guidance_scale=0,
            generator=generator,
            output_type="latent",
            s_churn=0,
            device=device,
            weight_dtype=weight_dtype,
            classifier=classifier,
            vae=vae,
            cell_subcelluar_location=cell_subcelluar_location,
            reference_channels_for_classifier=reference_channels_for_classifier
        )

        with torch.no_grad():
            # Decode the latents to images
            vae_type = vae.dtype
            vae.to(torch.float32)
            print(most_likely_indices)
            # Process each sample in the batch
            for sample_idx in range(effective_batch_size):
                sample_latents = generated_latents[sample_idx].to(torch.float32)
                generated_images_gt = decode_latents(vae, sample_latents[:,:16,:,:])
                
                protein_name = protein_names[sample_idx]
                cell_line_name = cell_line_names[sample_idx]
                
                print(f"Processing sample {sample_idx}: {cell_line_name}_{protein_name}")
                
                # Process all particles for this sample
                for i in range(sample_latents.shape[0]):
                    synthetic_image = generated_images_gt[i].cpu()
                    # average across dim 0 (channels)
                    synthetic_image = synthetic_image.mean(dim=0, keepdim=True)

                    # Save images for the first particle and best particle
                    if i == most_likely_indices[sample_idx]:
                        suffix = "best"
                        
                        # Fix: Use single index instead of slice to get 3D tensor, then add batch dim
                        cond_image_single = cond_images[sample_idx].cpu().numpy()  # Shape: [3, 256, 256]
                        synthetic_image_np = synthetic_image.numpy()  # Shape: [1, 256, 256]
                        
                        synthetic_stack = np.concatenate([synthetic_image_np, cond_image_single], axis=0)
                        synthetic_stack = synthetic_stack[[1, 0, 2, 3], :, :]
                        imwrite(
                            f"generated_images_2/{cell_line_name}_{protein_name}_pred_{suffix}_sample_{sample_idx}.tif",
                            synthetic_stack,
                        )

                        # Fix: Same issue with real_stack
                        gt_image_single = gt_images[sample_idx].cpu().numpy()  # Shape: [1, 256, 256]
                        
                        real_stack = np.concatenate([gt_image_single, cond_image_single], axis=0)
                        real_stack = real_stack[[1, 0, 2, 3], :, :]
                        imwrite(
                            f"generated_images_2/{cell_line_name}_{protein_name}_real_sample_{sample_idx}.tif",
                            real_stack,
                        )

                    # # Visualize comparison for the best particle
                    # if i == most_likely_indices[sample_idx]:
                    #     real_image = gt_images[sample_idx].cpu().float() * 0.5 + 0.5
                    #     print(f"Generated image {i} shape: {synthetic_image.shape}, Real image shape: {real_image.shape}")
                    #     # combine them side by side, but leave some space in between
                    #     combined_image = torch.cat([synthetic_image, real_image], dim=2)
                    #     plot_images(combined_image, row_title=f"generated_images_2/Generated Ground Truth Image_sample_{sample_idx}_particle_{i}")
            
            vae.to(vae_type)
        
        wandb.finish()
        
        # Break after first batch group for testing (remove this to process all data)
        break

    data = {
        "protein_name": protein_name_list,
        "cell_line_name": cell_line_name_list
    }

    df = pd.DataFrame(data)
    df.to_csv("protein_cell_line_names.csv", index=False)