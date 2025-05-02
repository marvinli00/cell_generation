import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

def create_unet_model(config=None, resolution=32):
    """
    Create a UNet model for conditional diffusion.
    
    Args:
        config: Optional model configuration.
        resolution: Image resolution expected by the model.
        
    Returns:
        UNet2DConditionModel: The configured UNet model.
    """
    if config is None:
        # # Create a default UNet configuration
        # model = UNet2DConditionModel(
        #     sample_size=32,
        #     in_channels=32,          # Total channels for combined input (gt + cond)
        #     out_channels=32,         # Output channels should match input dimension
        #     layers_per_block=2,
        #     block_out_channels=(256, 512, 512),
        #     down_block_types=(
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "AttnDownBlock2D",
        #     ),
        #     up_block_types=(
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "AttnUpBlock2D",
        #     ),
        #     cross_attention_dim = 768,
        #     mid_block_type="UNetMidBlock2DCrossAttn",
        #     # Add parameters for conditioning
        #     class_embed_type="simple_projection",
        #     projection_class_embeddings_input_dim=13348 + 40,  # Combined protein and cell line dimensions
        # )
        model = UNet2DConditionModel(
            sample_size=32,
            in_channels=32,          # Total channels for combined input (gt + cond)
            out_channels=32,         # Output channels should match input dimension
            layers_per_block=2,
            block_out_channels=(256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "AttnUpBlock2D",
            ),
            cross_attention_dim = 768,
            
            mid_block_type="UNetMidBlock2DCrossAttn",
            # Add parameters for conditioning
            class_embed_type="timestep",
            time_embedding_dim = 512,
            #num_class_embeds = 13348 + 40,
            #projection_class_embeddings_input_dim=13348 + 40,  # Combined protein and cell line dimensions
        )

        model.embedding_protein = nn.Embedding(13348,384)
        model.embedding_cell_label = nn.Embedding(40, 128)
    else:
        # Load configuration from file
        config = UNet2DConditionModel.load_config(config)
        model = UNet2DConditionModel.from_config(config)
    
    return model

def setup_model_devices(model, accelerator, weight_dtype, enable_xformers=False):
    """
    Set up model devices and optimizations.
    
    Args:
        model: The model to set up.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        enable_xformers: Whether to enable xformers for memory-efficient attention.
        
    Returns:
        model: The set up model.
    """
    # Move model to device and set weight type
    model = model.to(accelerator.device)
    model = model.to(weight_dtype)
    
    # Enable xformers if requested and available
    if enable_xformers:
        try:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ImportError("xformers is not available")
        except ImportError as e:
            print(f"Warning: {e}. xformers not enabled.")
    
    return model

def load_vae(vae_path, accelerator, weight_dtype):
    """
    Load a pre-trained VAE model.
    
    Args:
        vae_path: Path to the VAE model.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        
    Returns:
        vae: The loaded VAE model.
    """
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(accelerator.device)
    vae.to(weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    
    return vae

def load_classifier(checkpoint_path, accelerator, weight_dtype):
    """
    Load a pre-trained classifier model.
    
    Args:
        checkpoint_path: Path to the classifier checkpoint.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        
    Returns:
        classifier: The loaded classifier model.
    """
    from .location_classifier import LocationClassifier
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint['config']
    
    classifier = LocationClassifier(
        num_classes=config['num_classes'],
        pretrained=False,
        model_type=config['model_type']
    )
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    classifier.to(accelerator.device)
    classifier.to(weight_dtype)
    classifier.requires_grad_(False)
    
    return classifier

def load_clip_model(clip_model_path, accelerator, weight_dtype):
    """
    Load a pre-trained CLIP model.
    
    Args:
        clip_model_path: Path to the CLIP model.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        
    Returns:
        clip_model: The loaded CLIP model.
    """
    from .clip_image_encoder import OpenCLIPVisionEncoder
    
    clip_model = OpenCLIPVisionEncoder(
        model_name=clip_model_path,
        device=accelerator.device,
        weight_dtype=weight_dtype
    )
    clip_model.to(accelerator.device)
    clip_model.to(weight_dtype)
    clip_model.eval()
    clip_model.requires_grad_(False)
    return clip_model