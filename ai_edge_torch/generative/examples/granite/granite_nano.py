"""Example of building Granite Nano models."""

from typing import Callable, Dict, Tuple

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import loader as loading_utils
from ai_edge_torch.generative.utilities import model_builder
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    embedding="model.embed_tokens",
    ff_up_proj="model.layers.{}.shared_mlp.input_linear",
    ff_gate_proj="model.layers.{}.shared_mlp.gate_linear",
    ff_down_proj="model.layers.{}.shared_mlp.output_linear",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    final_norm="model.norm",
)

HYBRID_TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    embedding="model.embed_tokens",
    ff_up_proj="model.layers.{}.shared_mlp.input_linear",
    ff_gate_proj="model.layers.{}.shared_mlp.gate_linear",
    ff_down_proj="model.layers.{}.shared_mlp.output_linear",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    mamba_A_log="model.layers.{}.mamba.A_log",
    mamba_dt_bias="model.layers.{}.mamba.dt_bias",
    mamba_D="model.layers.{}.mamba.D",
    mamba_conv1d_weight="model.layers.{}.mamba.conv1d.weight",
    mamba_conv1d_bias="model.layers.{}.mamba.conv1d.bias",
    mamba_in_proj_weight="model.layers.{}.mamba.in_proj.weight",
    mamba_norm_weight="model.layers.{}.mamba.norm.weight",
    mamba_out_proj_weight="model.layers.{}.mamba.out_proj.weight",
    final_norm="model.norm",
)


def get_granite_nano_4_350m_config() -> cfg.ModelConfig:
  """Returns a ModelConfig approximating the IBM / HuggingFace Granite (nano) settings."""
  # RMSNorm used throughout (matches `normalization_function: "rmsnorm"` and rms_norm_eps=1e-05)
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-05,
      with_scale=False,
      use_bias=False,
      enable_hlfb=False,
  )

  # Attention config
  attn_config = cfg.AttentionConfig(
      num_heads=16,  # num_attention_heads
      head_dim=64,  # hidden_size / num_heads = 1024 / 16
      num_query_groups=4,
      # rotary_base mapped from rope_theta (large theta used by Granite). Using 10_000 default
      # but set to HF rope theta for better parity:
      rotary_base=10_000_000,  # maps to HF rope_theta: 10000000
      rotary_percentage=1.0,
      use_alibi=False,
      qkv_transpose_before_split=True,  # HF uses separate q/k/v then .view(...).transpose(1,2)
      qkv_use_bias=False,  # attention_bias = false in HF config
      qkv_fused_interleaved=False,
      output_proj_use_bias=False,
      enable_kv_cache=True,
      relative_attention_num_buckets=0,
      relative_attention_max_distance=0,
      attn_type=cfg.AttentionType.GLOBAL,
      sliding_window_size=None,
      attention_scale=0.015625,
  )

  # Feed-forward (Granite uses shared_intermediate_size / gated MLP with SILU gate)
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(type=cfg.ActivationType.SILU),
      intermediate_size=2048,  # HF: shared_intermediate_size / intermediate_size = 2048
      use_separate_gating=True,
      use_bias=False,
  )

  # Transformer block config — same block used for all 28 layers in this variant
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      parallel_residual=False,
      relative_attention=False,
      residual_multiplier=0.263,
  )

  config = cfg.ModelConfig(
      vocab_size=100352,  # HF: vocab_size
      num_layers=28,  # HF: num_hidden_layers
      max_seq_len=32768,  # HF: max_position_embeddings
      embedding_dim=1024,  # HF: hidden_size
      embedding_padding_idx=100256,
      block_configs=block_config,
      final_norm_config=norm_config,
      embedding_scale=12.0,  # HF: embedding_multiplier
      embedding_use_bias=False,
      image_embedding=None,
      num_mm_tokens_per_image=None,
      lm_head_use_bias=False,
      lm_head_share_weight_with_embedding=True,  # HF: tie_word_embeddings=true
      dense_intermediate_size=2048,  # shared intermediate size from HF
      enable_hlfb=True,
      final_logit_softcap=None,
      final_logit_scaling=4.0,
      # build_rope left default (rotary/rope behavior handled by code); attention_patterns left None
  )

  return config


def get_granite_nano_4h_350m_config() -> cfg.ModelConfig:
  """Returns a ModelConfig approximating the IBM / HuggingFace Granite (nano) settings."""
  # RMSNorm used throughout (matches `normalization_function: "rmsnorm"` and rms_norm_eps=1e-05)
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-05,
      with_scale=False,
      use_bias=False,
      enable_hlfb=False,
  )

  # Attention config
  attn_config = cfg.AttentionConfig(
      num_heads=12,  # num_attention_heads
      head_dim=64,  # hidden_size / num_heads = 768 / 12
      num_query_groups=4,
      rotary_percentage=1.0,
      use_alibi=False,
      qkv_transpose_before_split=True,  # HF uses separate q/k/v then .view(...).transpose(1,2)
      qkv_use_bias=False,  # attention_bias = false in HF config
      qkv_fused_interleaved=False,
      output_proj_use_bias=False,
      enable_kv_cache=True,
      relative_attention_num_buckets=0,
      relative_attention_max_distance=0,
      attn_type=cfg.AttentionType.GLOBAL,
      sliding_window_size=None,
      attention_scale=0.015625,
  )

  # Mamba config
  mamba_config = cfg.MambaConfig(
      hidden_size=768,
      chunk_size=256,
      d_conv=4,
      d_head=32,
      d_state=128,
      expand=2,
      n_groups=1,
      n_heads=48,
      norm_config=norm_config,
  )

  # Feed-forward (Granite uses shared_intermediate_size / gated MLP with SILU gate)
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(type=cfg.ActivationType.SILU),
      intermediate_size=2048,  # HF: shared_intermediate_size / intermediate_size = 2048
      use_separate_gating=True,
      use_bias=False,
  )

  attention_block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      parallel_residual=False,
      relative_attention=False,
      residual_multiplier=0.246,
  )
  mamba_block_config = cfg.TransformerBlockConfig(
      mamba_config=mamba_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      parallel_residual=False,
      relative_attention=False,
      residual_multiplier=0.246,
  )

  def dummy_build_rope(
      input_pos: torch.Tensor,
      n_elem: int,
      base: int = 10_000,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    return None

  config = cfg.ModelConfig(
      vocab_size=100352,  # HF: vocab_size
      num_layers=32,  # HF: num_hidden_layers
      max_seq_len=32768,  # HF: max_position_embeddings
      embedding_dim=768,  # HF: hidden_size
      embedding_padding_idx=100256,
      block_configs=[
          attention_block_config
          if i in {10, 13, 17, 27}
          else mamba_block_config
          for i in range(32)
      ],
      final_norm_config=norm_config,
      embedding_scale=12.0,  # HF: embedding_multiplier
      embedding_use_bias=False,
      image_embedding=None,
      num_mm_tokens_per_image=None,
      lm_head_use_bias=False,
      lm_head_share_weight_with_embedding=True,  # HF: tie_word_embeddings=true
      dense_intermediate_size=2048,  # shared intermediate size from HF
      enable_hlfb=True,
      final_logit_softcap=None,
      final_logit_scaling=3.0,
      build_rope=dummy_build_rope,  # hybrid model uses nope
  )

  return config


class Granite4Nano(model_builder.DecoderOnlyModel):
  """A Granite 4 Nano model built from the Edge Generative API layers."""

  pass


def granite_nano_custom_loader(checkpoint_path: str) -> Dict[str, torch.Tensor]:
  """Custom loader for Granite Nano that chunks input_linear.weight tensors.

  Loads safetensors and splits input_linear.weight of shape [2*output_dim, input_dim]
  into gate_linear.weight and input_linear.weight both of shape [output_dim, input_dim].
  """
  tensors = loading_utils.load_safetensors(checkpoint_path)

  # Process tensors to split input_linear.weight
  processed_tensors = {}
  for key, tensor in tensors.items():
    if "input_linear.weight" in key:
      # Chunk the tensor along dimension 0 (output dimension)
      output_dim = tensor.shape[0] // 2
      gate_weight = tensor[:output_dim, :]
      input_weight = tensor[output_dim:, :]

      # Create new keys
      gate_key = key.replace("input_linear.weight", "gate_linear.weight")
      processed_tensors[gate_key] = gate_weight
      processed_tensors[key] = input_weight
    else:
      processed_tensors[key] = tensor

  return processed_tensors


def build_nano4_350m_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_granite_nano_4_350m_config(),
      tensor_names=TENSOR_NAMES,
      model_class=Granite4Nano,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )


def build_nano4h_350m_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_granite_nano_4h_350m_config(),
      tensor_names=HYBRID_TENSOR_NAMES,
      model_class=Granite4Nano,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )
