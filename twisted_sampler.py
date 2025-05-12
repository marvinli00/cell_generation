import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.edm_utils import edm_clean_image_to_model_input, edm_model_output_to_x_0_hat

import math
from torch.distributions.normal import Normal
from functools import partial
import wandb
import tqdm
import wandb
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
    
    def normalize_weights(weights):
        return weights / torch.sum(weights)
    
    weights = normalize_weights(weights)
    
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
        tausq_ = tausq_ 
        return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
def compute_ess_softmax(log_weights):
    #softmax
    weights = torch.nn.functional.softmax(log_weights, dim = 0)
    return 1/torch.sum(weights**2)

import os
import torch
import torch.nn.functional as F
import numpy as np
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set precision (you can adjust this based on your hardware)
weight_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# # Define paths to models and checkpoints
# model_path = "/path/to/your/trained/model"  # Update this to your model checkpoint path
# vae_path = "/scratch/groups/emmalu/marvinli/twisted_diffusion/stable-diffusion-3.5-large-turbo/vae"
# classifier_path = "/scratch/groups/emmalu/marvinli/twisted_diffusion/checkpoints_classifier/model_epoch_7.pth"
# clip_model_path = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# Define paths to models and checkpoints
model_path = "/home/pc/Documents/twisted_diffusion/two_labels_latent_diffusion_edm_silu_less_cross_attn/checkpoint-200000"  # Update this to your model checkpoint path
vae_path = "/home/pc/Documents/twisted_diffusion_helper_model/vae"
classifier_path = "/home/pc/Documents/twisted_diffusion_helper_model/checkpoints_classifier/model_epoch_7.pth"
clip_model_path = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Set EDM parameters
sigma_min = EDM_CONFIG["SIGMA_MIN"]
sigma_max = EDM_CONFIG["SIGMA_MAX"]
sigma_data = EDM_CONFIG["SIGMA_DATA"]
rho = EDM_CONFIG["RHO"]

# Set sampling parameters
num_inference_steps = 1000
guidance_scale = 0  # Higher values increase adherence to the conditioning
batch_size = 1
image_size = 1  # Size of the generated images

# Create model
model = create_unet_model(resolution=image_size)
from diffusers import UNet2DConditionModel
model = CustomUNetWithEmbeddings.from_pretrained(model_path, subfolder="unet")
# # Load model checkpoint
# try:
#     # Try loading state dict directly
#     state_dict = torch.load(os.path.join(model_path, "unet", "diffusion_pytorch_model.bin"), map_location="cpu")
#     model.load_state_dict(state_dict)
# except:
#     # Fallback to loading from checkpoint file
#     checkpoint = torch.load(os.path.join(model_path, "checkpoint.pt"), map_location="cpu")
#     if "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     else:
#         model.load_state_dict(checkpoint)

# Move model to device and set to evaluation mode
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
    # Process CLIP image if provided
    if clip_image is not None:
        with torch.no_grad():
            encoder_hidden_states = clip_model(clip_image)
    else:
        # Create empty encoder hidden states
        encoder_hidden_states = torch.zeros((batch_size, 196, 768), device=device, dtype=weight_dtype)
    
    # Set up cell line and label conditioning
    if cell_line is None:
        # Create a one-hot vector for cell line (assuming 40 cell lines)
        cell_line = torch.zeros((batch_size, 40), device=device, dtype=weight_dtype)
        cell_line[:, 0] = 1.0  # Set first cell line as default
    
    if label is None:
        # Create a one-hot vector for label (assuming 13348 labels)
        label = torch.zeros((batch_size, 13348), device=device, dtype=weight_dtype)
        label[:, 0] = 1.0  # Set first label as default
    
    total_label = torch.cat([cell_line, label], dim=1)
    
    return encoder_hidden_states, total_label
    
    
def save_image(image, output_filename="output.png", **kwargs):
    """Save a single image to PNG file without using matplotlib"""
    from PIL import Image
    import numpy as np
    import torch
    
    # Handle torch tensor
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Handle different shapes and channel configurations
    if image.ndim == 4 and image.shape[0] == 1:  # [1, C, H, W]
        image = image[0]
    
    if image.shape[0] == 3 or image.shape[0] == 1:  # [C, H, W]
        image = image.transpose(1, 2, 0)
    
    # Handle single channel images
    if image.shape[-1] == 1:  # Single channel
        image = image.squeeze(-1)
        # For grayscale, just use L mode
        pil_mode = 'L'
    else:  # RGB
        pil_mode = 'RGB'
    
    # Ensure values are in valid range for PIL
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Create and save the PIL image
    if not output_filename.endswith('.png'):
        output_filename += '.png'
        
    pil_img = Image.fromarray(image, mode=pil_mode)
    pil_img.save(output_filename)
    
    return output_filename

def twisting_mse(x_0_hat, mask, sigma_dt, number_of_particles, step = 0):
    score = 0
    normalized_variance = get_xstart_var(sigma_dt, var_type = 1, tausq_ = 0.012)

    #Gaussian log probability: -||x-μ||²/(2σ²)
    mse_distance = (x_0_hat[:,16:,:,:] - mask) ** 2
    score_i = -torch.sum(mse_distance, dim=(1,2,3)) / (2*normalized_variance)
    
    #wandb.log({"mse_distance": mse_distance.mean()}, step=step)
    
    # Log the mean squared distance for monitoring/debugging
    #self.run.log({f"distances_of_motif": ((ts_com_zero - motif_target_cat[None,None,None,:])**2).mean()})
    #write to a
    # Add this motif's score to the total score
    # Multiple motifs' scores are summed, giving equal weight to each motif
    score = score + score_i
    #wandb.log({"softmax_score": torch.nn.functional.softmax(score, dim=0).max()}, step=step)
    
    #no need to logsumexp, because there is no permutation for mask
    #score_log_proob_given_motif = torch.logsumexp(score, dim=0) - torch.log(torch.tensor(number_of_particles, device=device))
    score_log_proob_given_motif = score - torch.log(torch.tensor(number_of_particles, device=device))
    
    return score_log_proob_given_motif, torch.argmax(score)


def sample_edm(
    model,
    scheduler,
    twisting_target,
    batch_size=1,
    image_size=32,
    number_of_particles=16,
    
    num_inference_steps=50,
    condition_latent=None,  # Optional conditioning latent
    encoder_hidden_states=None,  # CLIP hidden states
    class_labels=None,  # Class labels for conditioning
    
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
):      
    wandb.init(project="Twisted Diffusion", name="Twisted Diffusion")
    batch_size = number_of_particles
    protein_labels = protein_labels.repeat(batch_size, 1)
    cell_line_labels = cell_line_labels.repeat(batch_size, 1)
    encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)
    
    if classifier is not None:
        cell_subcelluar_location = cell_subcelluar_location.repeat(batch_size, 1).to(weight_dtype)
    
    # Create random noise for the ground truth part
    latent_channels = 32
    gt_noise = torch.randn(
        (batch_size, latent_channels, image_size, image_size),
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
    #ess_tracker.append(
    
    #initialize weights
    log_w_prev_accumulated = torch.log(torch.ones_like(log_proposal, device = device))
    

    # Define steps
    
    
    torch_default_dtype = torch.get_default_dtype()

    # Iterate
    for i, t in enumerate(progress_bar):

        sigma = scheduler.sigmas[i]
        sigma_next = scheduler.sigmas[i + 1] if i < len(scheduler.sigmas) - 1 else torch.tensor(0.0, device=device)
        
        # Expand sigma for broadcasting
        sigma_expanded = sigma.expand(batch_size).to(device)
        sigma_view = sigma_expanded.view(-1, 1, 1, 1).double()
        
        # Calculate gamma for stochastic sampling (from the step method)
        gamma = min(s_churn / (len(scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        sigma_hat = sigma * (gamma + 1)
        sigma_hat_view = sigma_hat.view(-1, 1, 1, 1).double().to(device)

        
        # Add noise if gamma > 0 (s_churn is active) - implements stochastic sampling
        if gamma > 0:
            noise = torch.randn((batch_size, latent_channels, image_size, image_size), generator=generator, device=device, dtype=latents.dtype)
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
        model.to(weight_dtype)
        model_input = model_input.to(weight_dtype)
        timestep_input = timestep_input.to(weight_dtype)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
        
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
            predicted_class = classifier(untwisted_predicted_x_start*4/vae.scaling_factor)
            probs = torch.sigmoid(predicted_class)
            eps = 1e-5
            log_prob_classifier = torch.log(probs + eps) * cell_subcelluar_location + torch.log(1 - probs + eps) * (1 - cell_subcelluar_location)
            log_prob_classifier = log_prob_classifier.sum(dim = 1)
            most_likely_index = torch.argmax(log_prob_classifier, dim=0)
            wandb.log({"log_prob_classifier": log_prob_classifier.mean()}, step=i)
            wandb.log({"most_likely_index": most_likely_index}, step=i)

            
            
        
        #compute log p(y|x_t, t) := log N(y; x_0, sigma_t^2 I)
        log_prob, most_likely_index = twisting_mse(untwisted_predicted_x_start, twisting_target, step_sigma, number_of_particles, step = i)
        log_prob = log_prob.squeeze()
        
        log_prob_classifier = log_prob_classifier.squeeze()
        log_prob = log_prob + log_prob_classifier
        
        grad_pk_with_respect_to_x_t = torch.autograd.grad(log_prob.mean(), combined_latent)[0]
        #rescale mean back to the original scale
        grad_pk_with_respect_to_x_t = grad_pk_with_respect_to_x_t*combined_latent.shape[0]
        if i < 500:   
            with torch.no_grad():
                #alpha = 0.05
                alpha = 0.012
                # |grad_pk_with_respect_to_x_t|_F
                norm_grad = grad_pk_with_respect_to_x_t.norm()
                #regularize gradient to prevent gradient explosion
            grad_pk_with_respect_to_x_t = grad_pk_with_respect_to_x_t*alpha*norm_grad/(alpha+norm_grad)
         
        else:
            with torch.no_grad():
                #alpha = 0.05
                alpha = 0.012
                # |grad_pk_with_respect_to_x_t|_F
                norm_grad = grad_pk_with_respect_to_x_t.norm()
                #regularize gradient to prevent gradient explosion
                grad_pk_with_respect_to_x_t = grad_pk_with_respect_to_x_t*alpha*norm_grad/(alpha+norm_grad)
        twisted_predicted_x_start = untwisted_predicted_x_start + grad_pk_with_respect_to_x_t
        
        # #find u(x_t-1) = E[x_t-1|x_t, (x_0, y),t]
        # trans_mean = (posterior_mean_coef1 * twisted_predicted_x_start + 
        #                 posterior_mean_coef2 * ts.trans)
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
                
            ess =  compute_ess_from_log_w(log_w_accumulated)
            wandb.log({"ess": ess}, step=i)
            # ess = self.compute_ess(log_w_accumulated)
            ess_tracker.append(ess.detach().cpu().numpy())
            #self.run.log({"ess": ess})
            #resample when ess is too low (50% of num_samples)
            if ess < 0.5*number_of_particles:
                weights = torch.nn.functional.softmax(log_w_accumulated, dim = 0)
                #resample
                latents_twisted, log_w_prev_accumulated, indexes = systematic_resampling(latents_twisted, weights)
                log_proposal_tracker[0] = log_proposal_tracker[0][indexes]
            else:
                #log_w = normalize_log_weights(log_w, dim=0)
                log_w_prev_accumulated = normalize_log_weights(log_w_accumulated, dim=0) + torch.log(torch.tensor(number_of_particles, device=device))
                # Compute rotations
            latents = latents_twisted
    return latents, most_likely_index

test_dataset = FullFieldDataset(
        data_root='/home/pc/Documents/twisted_diffusion_helper_model/test_images',
        label_dict='/home/pc/Documents/twisted_diffusion_helper_model/antibody_map.pkl',
        annotation_dict='/home/pc/Documents/twisted_diffusion_helper_model/annotation_map.pkl',
        is_train=False
    )
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False,
)
# Get a batch of test data
#batch = next(iter(test_dataloader))
from tqdm import tqdm
    # Generate samples
weight_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
clip_model.to(weight_dtype)
count = 0
for batch in tqdm(test_dataloader):

    # Show conditioning image
    cond_images = batch["cond_image"].to(weight_dtype).to(device)
    clip_images = batch["clip_image"].to(weight_dtype).to(device)
    gt_images = batch["gt_image"].to(weight_dtype).to(device)
    # Encode conditioning image to latent space
    with torch.no_grad():
        #cond_images_latent = prepare_latent_sample(vae, cond_images.repeat(1, 3, 1, 1), weight_dtype)
        encoder_hidden_states = clip_model(clip_images)
        
    # Prepare cell_line and label conditioning
    cell_line = batch["cell_line"].to(device).long()
    protein_label = batch["label"].to(device).long()
    
    cell_subcelluar_location = batch["annotation"].to(device).long()
    
    #one hot encoding
    #cell_line = torch.nn.functional.one_hot(cell_line, num_classes=40)
    #label = torch.nn.functional.one_hot(label, num_classes=13348)

    #total_label = torch.cat([cell_line, label], dim=1)

    
    

    num_inference_steps=300

    # Set random seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)
    twisting_target = cond_images
    
    twisting_target = prepare_latent_sample(vae, twisting_target, weight_dtype)*vae.scaling_factor/4
    
    # Sample from the model
    generated_latents, most_likely_latent_index = sample_edm(
        model=model,
        scheduler=scheduler,
        batch_size=1,
        number_of_particles=16,
        twisting_target = twisting_target,
        image_size=32,
        num_inference_steps=num_inference_steps,
        condition_latent=None,
        encoder_hidden_states=encoder_hidden_states,
        class_labels=None,
        protein_labels = protein_label,
        cell_line_labels = cell_line,
        guidance_scale=0,
        generator=generator,
        output_type="latent",
        s_churn = 0,
        device = device,
        weight_dtype = weight_dtype,
        classifier = classifier,
        vae = vae,
        cell_subcelluar_location = cell_subcelluar_location
    )
    with torch.no_grad():
        # Decode the latents to images
        vae_type = vae.dtype
        vae.to(torch.float32)
        generated_latents = generated_latents.to(torch.float32)
        generated_images_gt = decode_latents(vae, generated_latents[:,:16,:,:])
        generated_images_cond = decode_latents(vae, generated_latents[:,16:,:,:])
        vae.to(vae_type)
    
        # Display conditioning image
        
    
    # save_image(cond_images[0].cpu().float()*0.5+0.5, output_filename=f"generated_images/Testset_Conditioning Image_{count}")
    # save_image(generated_images_gt[most_likely_latent_index].cpu(), output_filename=f"generated_images/Generated Ground Truth Image_{count}")
    # save_image(gt_images[0].cpu().float()*0.5+0.5, output_filename=f"generated_images/Testset_Ground Truth Image_{count}")
    # save_image(generated_images_cond[most_likely_latent_index].cpu(), output_filename=f"generated_images/Generated Conditioning Image_{count}")
    # count += 1
    
    for i in range(generated_latents.shape[0]):
        save_image(cond_images[0].cpu().float()*0.5+0.5, output_filename=f"generated_images/Testset_Conditioning Image_{count}")
        save_image(generated_images_gt[i].cpu(), output_filename=f"generated_images/Generated Ground Truth Image_{count}")
        save_image(gt_images[0].cpu().float()*0.5+0.5, output_filename=f"generated_images/Testset_Ground Truth Image_{count}")
        save_image(generated_images_cond[i].cpu(), output_filename=f"generated_images/Generated Conditioning Image_{count}")
        count += 1
        
    
