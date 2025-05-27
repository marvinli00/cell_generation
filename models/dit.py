# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import register_to_config, ConfigMixin

from diffusers.utils import is_torch_version, logging
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel 
from models.MarvinsBasicTransformerBlock import MarvinsBasicTransformerBlock
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiTTransformer2DModelWithCrossAttention(DiTTransformer2DModel):
    r"""
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        cross_attention_dim = None,
        #must couple with 
        num_protein_labels = 13349,  # 13348 + 1
        num_cell_labels = 41,        # 40 + 1,
        positional_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            upcast_attention=upcast_attention,
            norm_type=norm_type if norm_type != "ada_norm_zero_continuous" else "ada_norm_zero",
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            # The DiTTransformer2DModelWithCrossAttention is a subclass of DiTTransformer2DModel
        )
        # self.patch_size = self.config.patch_size
        # self.pos_embed = PatchEmbed(
        #     height=self.config.sample_size,
        #     width=self.config.sample_size,
        #     patch_size=self.config.patch_size,
        #     in_channels=self.config.in_channels,
        #     embed_dim=self.inner_dim,
        # )

        self.cross_attention_dim = cross_attention_dim
        self.proj_out_1 = nn.Linear(2 * self.inner_dim, 2 * self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                MarvinsBasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    cross_attention_dim = self.cross_attention_dim,
                    num_protein_labels = num_protein_labels,
                    num_cell_labels = num_cell_labels,
                    positional_embeddings=positional_embeddings,
                )   
                for _ in range(self.config.num_layers)
            ]
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        protein_labels: Optional[torch.Tensor] = None,
        cell_line_labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        #class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    protein_labels,
                    cell_line_labels
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    #class_labels=class_labels,
                    protein_labels=protein_labels,
                    cell_line_labels=cell_line_labels,
                )

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.get_emb(timestep, protein_labels, cell_line_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        # unpatchify
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def create_dit_model(config=None, resolution=32):
    if config is None:
        model = DiTTransformer2DModelWithCrossAttention(
                num_attention_heads = 16,
                attention_head_dim = 72,
                in_channels = 16,
                out_channels = 16,
                num_layers = 28,
                dropout = 0.0,
                norm_num_groups = 32,
                attention_bias = True,
                sample_size = 32,
                patch_size = 2,
                activation_fn = "gelu-approximate",
                num_embeds_ada_norm = 1000,
                upcast_attention = False,
                norm_type = "ada_norm_zero_continuous",
                norm_elementwise_affine = False,
                norm_eps = 1e-5,
                cross_attention_dim = 768,
                #must couple with ada_norm_zero_continuous
                num_protein_labels = 13349,  # 13348 + 1
                num_cell_labels = 41,        # 40 + 1,
                positional_embeddings = None
            )

    else:
        print("no model")
    
    return model