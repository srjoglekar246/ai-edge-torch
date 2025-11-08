# Copyright 2024 The AI Edge Torch Authors.
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
# ==============================================================================

"""Common building blocks for Attention layer."""

import abc
from typing import Optional, Tuple, Union

from ai_edge_torch.generative.layers import attention_utils
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import lora as lora_utils
from ai_edge_torch.generative.layers import mamba_cache as mamba_utils
from ai_edge_torch.generative.layers import scaled_dot_product_attention as sdpa
from ai_edge_torch.generative.layers import sdpa_with_kv_update
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
import torch
from torch import nn


class TransformerBlock(nn.Module):

  def __init__(
      self,
      config: cfg.TransformerBlockConfig,
      model_config: cfg.ModelConfig,
  ) -> None:
    """Initialize an instance of the TransformerBlock.

    Args:
      config (cfg.TransformerBlockConfig): the configuration object for this
        transformer block.
      model_config (cfg.ModelConfig): the configuration object for the model
        this transformer block belongs to.
    """
    super().__init__()
    self.pre_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.pre_attention_norm_config,
    )
    self.atten_func = None
    self.mamba_func = None
    if config.attn_config is not None:
      self.atten_func = CausalSelfAttention(
          model_config.embedding_dim,
          config.attn_config,
          model_config.enable_hlfb,
      )
    elif config.mamba_config is not None:
      self.mamba_func = MambaLayer(
          config.mamba_config,
      )
    self.post_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.post_attention_norm_config,
    )
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)
    self.config = config

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: kv_utils.KVCacheEntry = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
      mamba_cache: Optional[mamba_utils.MambaCacheEntry] = None,
      mamba_mask: Optional[torch.Tensor] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the TransformerBlock.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the optional kv cache entry.
      lora (LoRAEntry): the optional lora entry.
      mamba_cache (MambaCacheEntry): the optional mamba cache entry.

    Returns:
      output activation from this transformer block, and updated kv cache (if
      passed in).
    """
    kv = None
    if self.config.parallel_residual:
      x_norm = self.pre_atten_norm(x)
      if self.atten_func is not None:
        atten_func_out = self.atten_func(
            x_norm, rope, mask, input_pos, kv_cache, lora
        )
      else:
        atten_func_out = self.mamba_func(
            x_norm, mask, mamba_cache=mamba_cache, mamba_mask=mamba_mask
        )
      if kv_cache is None and mamba_cache is None:
        attn_out = atten_func_out
      elif kv_cache is None:
        attn_out, mamba_cache = atten_func_out
      elif mamba_cache is None:
        attn_out, kv = atten_func_out
      ff_out = self.ff(x_norm)
      output = x + (attn_out + ff_out) * self.config.residual_multiplier
    else:
      x_norm = self.pre_atten_norm(x)
      if self.atten_func is not None:
        atten_func_out = self.atten_func(
            x_norm, rope, mask, input_pos, kv_cache, lora
        )
      else:
        atten_func_out = self.mamba_func(
            x_norm, mamba_cache=mamba_cache, mamba_mask=mamba_mask
        )
      if kv_cache is None and mamba_cache is None:
        attn_out = atten_func_out
        cache_out = None
      elif kv_cache is None:
        attn_out, cache_out = atten_func_out
      elif mamba_cache is None:
        attn_out, cache_out = atten_func_out
      x = x + attn_out * self.config.residual_multiplier
      x_norm = self.post_atten_norm(x)
      output = x + self.ff(x_norm) * self.config.residual_multiplier

    return output if cache_out is None else (output, cache_out)


class CausalSelfAttentionBase(nn.Module):
  """Base class for causal self attention layer."""

  def __init__(
      self, dim: int, config: cfg.AttentionConfig, enable_hlfb: bool
  ) -> None:
    super().__init__()
    self.dim = dim
    self.config = config
    self.enable_hlfb = enable_hlfb

    self.query_norm = builder.build_norm(
        self.config.head_dim, self.config.query_norm_config
    )
    self.key_norm = builder.build_norm(
        self.config.head_dim, self.config.key_norm_config
    )
    self.value_norm = builder.build_norm(
        self.config.head_dim, self.config.value_norm_config
    )

  @abc.abstractmethod
  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    raise NotImplementedError()


class CausalSelfAttention(CausalSelfAttentionBase):
  """Causal self attention layer implementation."""

  def __init__(
      self,
      dim: int,
      config: cfg.AttentionConfig,
      enable_hlfb: bool,
  ) -> None:
    """Initialize an instance of CausalSelfAttention.

    Args:
      dim (int): causal attention's input/output dimmension.
      config (cfg.AttentionConfig): attention specific configurations.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__(dim, config, enable_hlfb)
    self.kv_cache = None
    qkv_shape = (
        config.num_heads + 2 * config.num_query_groups
    ) * config.head_dim
    output_shape = config.num_heads * config.head_dim
    # Key, query, value projections for all heads.
    self.qkv_projection = nn.Linear(dim, qkv_shape, bias=config.qkv_use_bias)
    self.output_projection = nn.Linear(
        output_shape, dim, bias=config.output_proj_use_bias
    )

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the CausalSelfAttention layer, which can support

       MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
      lora (LoRAEntry): the optional lora entry.

    Returns:
      output activation from this self attention layer, and the updated
        KV Cach Entry (if passed in).
    """
    # Batch size, sequence length, embedding dimensionality.
    B, T, _ = x.size()
    qkv = self.qkv_projection(x)

    # Assemble into a number of query groups to support MHA, MQA and GQA.
    q_per_kv = self.config.num_heads // self.config.num_query_groups
    # Each group has >=1 queries, 1 key, and 1 value.
    if self.config.qkv_transpose_before_split:
      qkv = qkv.view(B, T, -1, self.config.head_dim)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.num_query_groups,
              self.config.num_query_groups,
              self.config.num_query_groups,
          ),
          dim=-2,
      )
    else:
      qkv = qkv.view(B, T, self.config.num_query_groups, -1)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.head_dim,
              self.config.head_dim,
              self.config.head_dim,
          ),
          dim=-1,
      )

    if lora is not None:
      q += lora_utils.apply_lora(x, lora.attention.query, shape=q.shape)
      k += lora_utils.apply_lora(x, lora.attention.key, shape=k.shape)
      v += lora_utils.apply_lora(x, lora.attention.value, shape=v.shape)

    q = self.query_norm(q)
    k = self.key_norm(k)
    v = self.value_norm(v)

    q = q.reshape(B, T, -1, self.config.head_dim)
    k = k.reshape(B, T, -1, self.config.head_dim)
    v = v.reshape(B, T, -1, self.config.head_dim)

    alibi_bias = None
    if self.config.use_alibi:
      k_size = T
      if mask is not None:
        k_size = mask.shape[-1]
      elif input_pos is not None:
        # If mask is not present, assume current sequence length is key length.
        k_size = input_pos[-1].item() + 1
      alibi_bias = attention_utils.build_alibi_bias(
          n_heads=self.config.num_heads,
          k_size=k_size,
          dtype=x.dtype,
          device=x.device,
      )
    elif rope is not None:
      # Compute rotary positional embedding for query and key.
      cos, sin = rope
      q, k = rotary_pos_emb.apply_rope_inline(q, k, cos, sin)

    sdpa_out, kv_cache = sdpa_with_kv_update.sdpa_with_kv_update(
        q,
        k,
        v,
        kv_cache,
        input_pos,
        mask,
        self.config,
        self.enable_hlfb,
        alibi_bias=alibi_bias,
    )

    # Compute the output projection.
    y = self.output_projection(sdpa_out)
    if lora is not None:
      y += lora_utils.apply_lora(sdpa_out, lora.attention.output)

    return y if kv_cache is None else (y, kv_cache)


class SelfAttention(CausalSelfAttention):
  """Non-causal Self Attention module, which is equivalent to CausalSelfAttention without mask."""

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the SelfAttention layer, which can support MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
      lora (LoRAEntry): the optional lora entry.

    Returns:
      output activation from this self attention layer, and the updated
        KV Cach Entry (if passed in).
    """
    B, T, _ = x.size()
    assert (
        kv_cache is None
    ), "KV cache is not supported in non-causal SelfAttention."
    return super().forward(
        x,
        rope=rope,
        mask=torch.zeros((B, 1, T, T), dtype=torch.float32),
        input_pos=input_pos,
        lora=lora,
    )


class CrossAttention(nn.Module):

  def __init__(
      self,
      query_dim: int,
      cross_dim: int,
      hidden_dim: int,
      output_dim: int,
      config: cfg.AttentionConfig,
      enable_hlfb: bool,
  ):
    """Initialize an instance of CrossAttention.

    Args:
      query_dim (int): query tensor's dimension.
      cross_dim (int): cross attention's dimensions, for key and value tensors.
      hidden_dim (int): hidden dimension that q, k, v tensors project to.
      output_dim (int): output tensor's dimension.
      config (cfg.AttentionConfig): attention specific configurations.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__()
    self.config = config
    self.n_heads = config.num_heads
    self.q_projection = nn.Linear(
        query_dim, hidden_dim, bias=config.qkv_use_bias
    )
    self.k_projection = nn.Linear(
        cross_dim, hidden_dim, bias=config.qkv_use_bias
    )
    self.v_projection = nn.Linear(
        cross_dim, hidden_dim, bias=config.qkv_use_bias
    )
    self.output_projection = nn.Linear(
        hidden_dim, output_dim, bias=config.output_proj_use_bias
    )

    self.sdpa_func = (
        sdpa.scaled_dot_product_attention_with_hlfb
        if enable_hlfb
        else sdpa.scaled_dot_product_attention
    )

  def forward(
      self,
      x: torch.Tensor,
      y: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ):
    """Forward function of the CrossAttention layer.

    Args:
      x (torch.Tensor): the target tensor, with shape [B, target_seq_len, ...].
      y (torch.Tensor): the source tensor, with shape [B, source_seq_len, ...].
      rope (Tuple[torch.Tensor, torch.Tensor]): the optional input rope tensor.
      mask (torch.Tensor): the optional mask tensor can be broadcaseted to shape
        [B, n_heads, target_seq_len, source_seq_len].
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
      lora (LoRAEntry): the optional lora entry.

    Returns:
      output activation from this cross attention layer.
    """
    batch_size = x.size()[0]
    target_seq_len = x.size()[1]
    source_seq_len = y.size()[1]

    q = self.q_projection(x)
    k = self.k_projection(y)
    v = self.v_projection(y)

    if lora is not None:
      q += lora_utils.apply_lora(x, lora.attention.query, shape=q.shape)
      k += lora_utils.apply_lora(x, lora.attention.key, shape=k.shape)
      v += lora_utils.apply_lora(x, lora.attention.value, shape=v.shape)

    interim_shape = (batch_size, -1, self.n_heads, self.config.head_dim)
    q = q.view(interim_shape)
    k = k.view(interim_shape)
    v = v.view(interim_shape)

    if rope is not None:
      # Compute rotary positional embedding for query and key.
      cos, sin = rope
      q, k = rotary_pos_emb.apply_rope_inline(q, k, cos, sin)

    if kv_cache is not None:
      kv_cache = kv_utils.update(kv_cache, input_pos, k, v)
      k, v = kv_cache.k_cache, kv_cache.v_cache
    if mask is None:
      mask = torch.zeros(
          (batch_size, 1, target_seq_len, source_seq_len), dtype=torch.float32
      )
    y = self.sdpa_func(q, k, v, self.config.head_dim, mask=mask)
    y = y.reshape(batch_size, target_seq_len, -1)

    # Compute the output projection.
    y = self.output_projection(y)
    if lora is not None:
      y += lora_utils.apply_lora(y, lora.attention.output)

    return y if kv_cache is None else (y, kv_cache)


def apply_mask_to_padding_states(
    input_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
  """Apply attention mask to input states by zeroing out padded positions."""
  if attention_mask is not None:
    # Expand mask to match input_states dimensions
    mask = attention_mask.unsqueeze(-1).expand_as(input_states)
    input_states = input_states * mask
  return input_states


def pad_tensor_by_size(tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
  """Pad tensor along the sequence dimension."""
  if pad_size == 0:
    return tensor
  padding = (0, 0, 0, 0, 0, pad_size)  # Pad last dimension
  return torch.nn.functional.pad(tensor, padding)


def reshape_into_chunks(
    tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
  """Reshape tensor into chunks for processing."""
  tensor = pad_tensor_by_size(tensor, pad_size)
  batch_size, seq_len = tensor.shape[:2]
  chunk_size = min(seq_len, chunk_size)
  return tensor.reshape(
      batch_size, seq_len // chunk_size, chunk_size, *tensor.shape[2:]
  )


def segment_sum(input_tensor):
  """
  More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
  """
  chunk_size = input_tensor.size(-1)
  # 1. expand input tensor to have an additional dimension and repeat along that dimension
  # [..., chunk_size] -> [..., chunk_size, chunk_size]
  input_tensor = input_tensor[..., None].expand(
      *input_tensor.size(), chunk_size
  )
  # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
  mask = torch.tril(
      torch.ones(
          chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
      ),
      diagonal=-1,
  )
  input_tensor = input_tensor.masked_fill(~mask, 0)
  # 3. compute actual cumsum
  tensor_segsum = torch.cumsum(input_tensor, dim=-2)

  # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
  mask = torch.tril(
      torch.ones(
          chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
      ),
      diagonal=0,
  )
  tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
  return tensor_segsum


class MambaLayer(nn.Module):
  """MambaV2 layer implementation for selective state space models."""

  def __init__(
      self,
      mamba_config: cfg.MambaConfig,
  ) -> None:
    """Initialize MambaLayer.

    Args:
      dim: Input/output dimension
      mamba_config: Mamba-specific configuration
    """
    super().__init__()

    dim = mamba_config.hidden_size
    self.mamba_config = mamba_config
    self.intermediate_size = int(mamba_config.expand * mamba_config.hidden_size)
    self.conv_dim = (
        self.intermediate_size
        + 2 * mamba_config.n_groups * mamba_config.d_state
    )

    # Time step limits
    self.time_step_limit = (0.0, float("inf"))

    # Convolution layer
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=mamba_config.conv_bias,
        kernel_size=mamba_config.d_conv,
        groups=self.conv_dim,
        padding=mamba_config.d_conv - 1,
    )

    # Input projection
    projection_size = (
        self.intermediate_size + self.conv_dim + mamba_config.n_heads
    )
    self.in_proj = nn.Linear(dim, projection_size, bias=mamba_config.proj_bias)

    # Time step bias
    self.dt_bias = nn.Parameter(torch.ones(mamba_config.n_heads))

    # State space parameters
    A = torch.arange(1, mamba_config.n_heads + 1)
    self.A_log = nn.Parameter(torch.log(A))
    self.D = nn.Parameter(torch.ones(mamba_config.n_heads))

    self.norm = builder.build_norm(
        self.intermediate_size,
        mamba_config.norm_config,
    )

    # Output projection
    self.out_proj = nn.Linear(
        self.intermediate_size, dim, bias=mamba_config.proj_bias
    )

  def forward(
      self,
      x: torch.Tensor,
      mamba_cache: Optional[mamba_utils.MambaCacheEntry] = None,
      mamba_mask: Optional[torch.Tensor] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, mamba_utils.MambaCacheEntry]]:
    """Forward pass of MambaLayer."""
    batch_size, seq_len, _ = x.shape
    dtype = x.dtype
    # chunk_size = min(seq_len, self.mamba_config.chunk_size)
    chunk_size = seq_len

    silu = torch.nn.functional.silu

    # Apply attention mask to input
    x = apply_mask_to_padding_states(x, mamba_mask)

    # Input projection
    projected_states = self.in_proj(x)
    gate, hidden_states_B_C, dt = projected_states.split(
        [self.intermediate_size, self.conv_dim, self.mamba_config.n_heads],
        dim=-1,
    )

    # Convolution transformation with caching
    if seq_len == 1 and mamba_cache is not None:
      # Update conv state for single token by slicing and concatenating
      new_conv_input = hidden_states_B_C[:, 0, :]  # [batch_size, conv_dim]
      # Slice out old states and concatenate with new input
      old_conv_states = mamba_cache.conv_state[:, :, 1:]  # Remove oldest state
      new_conv_state = torch.cat(
          [old_conv_states, new_conv_input.unsqueeze(-1)], dim=-1
      )

      # Compute convolution output using new state
      conv_weights = self.conv1d.weight.squeeze(1)
      hidden_states_B_C = torch.sum(new_conv_state * conv_weights, dim=-1)
      if self.conv1d.bias is not None:
        hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
      hidden_states_B_C = silu(hidden_states_B_C)
      hidden_states_B_C = hidden_states_B_C.unsqueeze(1)
    else:
      # Standard convolution for prefill or non-cached inference
      new_conv_state = None
      if mamba_cache is not None:
        # Initialize cache with padded input
        hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
        new_conv_state = nn.functional.pad(
            hidden_states_B_C_transposed,
            (
                self.mamba_config.d_conv
                - hidden_states_B_C_transposed.shape[-1],
                0,
            ),
        )

      hidden_states_B_C = silu(
          self.conv1d(hidden_states_B_C.transpose(1, 2))[
              ..., :seq_len
          ].transpose(1, 2)
      )

    # Apply mask again after convolution
    hidden_states_B_C = apply_mask_to_padding_states(
        hidden_states_B_C, mamba_mask
    )

    # Split hidden states, B, and C
    hidden_states, B, C = torch.split(
        hidden_states_B_C,
        [
            self.intermediate_size,
            self.mamba_config.n_groups * self.mamba_config.d_state,
            self.mamba_config.n_groups * self.mamba_config.d_state,
        ],
        dim=-1,
    )

    # SSM transformation
    A = -torch.exp(self.A_log.float())

    if seq_len == 1 and mamba_cache is not None:
      # Single token generation path
      dt = dt[:, 0, :][:, None, ...]
      dt = dt.transpose(1, 2).expand(
          batch_size, dt.shape[-1], self.mamba_config.d_head
      )
      dt_bias = self.dt_bias[..., None].expand(
          self.mamba_config.n_heads, self.mamba_config.d_head
      )

      dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
      dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

      A_expanded = (
          A[..., None, None]
          .expand(
              self.mamba_config.n_heads,
              self.mamba_config.d_head,
              self.mamba_config.d_state,
          )
          .to(dtype=torch.float32)
      )
      dA = torch.exp(dt[..., None] * A_expanded)

      # Discretize B
      B = B.reshape(batch_size, self.mamba_config.n_groups, -1)[..., None, :]
      B = B.expand(
          batch_size,
          self.mamba_config.n_groups,
          self.mamba_config.n_heads // self.mamba_config.n_groups,
          B.shape[-1],
      ).contiguous()
      B = B.reshape(batch_size, -1, B.shape[-1])
      dB = dt[..., None] * B[..., None, :]

      # Hidden states to proper shape
      hidden_states = hidden_states.reshape(
          batch_size, -1, self.mamba_config.d_head
      )
      dBx = dB * hidden_states[..., None]

      # Update SSM state
      new_ssm_state = mamba_cache.ssm_state * dA + dBx

      # Compute output
      C = C.reshape(batch_size, self.mamba_config.n_groups, -1)[..., None, :]
      C = C.expand(
          batch_size,
          self.mamba_config.n_groups,
          self.mamba_config.n_heads // self.mamba_config.n_groups,
          C.shape[-1],
      ).contiguous()
      C = C.reshape(batch_size, -1, C.shape[-1])

      ssm_states_reshaped = new_ssm_state.view(
          batch_size * self.mamba_config.n_heads,
          self.mamba_config.d_head,
          self.mamba_config.d_state,
      )
      C_reshaped = C.view(
          batch_size * self.mamba_config.n_heads, self.mamba_config.d_state, 1
      )
      y = torch.bmm(ssm_states_reshaped, C_reshaped)
      y = y.view(
          batch_size, self.mamba_config.n_heads, self.mamba_config.d_head
      )

      # D skip connection
      D = self.D[..., None].expand(
          self.mamba_config.n_heads, self.mamba_config.d_head
      )
      y = y + hidden_states * D
      y = y.reshape(batch_size, -1)[:, None, ...]

      # Update cache with new states
      mamba_cache = mamba_utils.update(
          mamba_cache, new_conv_state, new_ssm_state
      )
    else:
      # Prefill path (chunked processing)
      dt = nn.functional.softplus(dt + self.dt_bias)
      dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

      hidden_states = hidden_states.reshape(
          batch_size, seq_len, -1, self.mamba_config.d_head
      ).float()
      B = B.reshape(batch_size, seq_len, -1, self.mamba_config.d_state).float()
      C = C.reshape(batch_size, seq_len, -1, self.mamba_config.d_state).float()

      B = B.repeat_interleave(
          self.mamba_config.n_heads // self.mamba_config.n_groups,
          dim=2,
          output_size=self.mamba_config.n_heads,
      )
      C = C.repeat_interleave(
          self.mamba_config.n_heads // self.mamba_config.n_groups,
          dim=2,
          output_size=self.mamba_config.n_heads,
      )

      pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
      D_residual = self.D[..., None] * pad_tensor_by_size(
          hidden_states, pad_size
      )

      # Discretize
      hidden_states = hidden_states * dt[..., None]
      A_dt = A.to(hidden_states.dtype) * dt

      # Reshape into chunks
      # hidden_states, A_dt, B, C = [
      #     reshape_into_chunks(t, pad_size, chunk_size)
      #     for t in (hidden_states, A_dt, B, C)
      # ]
      hidden_states, A_dt, B, C = [
          x.unsqueeze(1) for x in (hidden_states, A_dt, B, C)
      ]

      A_dt = A_dt.permute(0, 3, 1, 2)
      A_cumsum = torch.cumsum(A_dt, dim=-1)

      # Intra-chunk processing
      L = torch.exp(segment_sum(A_dt))
      G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
      G = G_intermediate.sum(dim=-1)
      M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
      M = M_intermediate.sum(dim=-1)
      Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

      # Inter-chunk processing
      decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
      B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
      states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

      previous_states = torch.zeros_like(states[:, :1])
      if mamba_cache is not None:
        previous_states = mamba_cache.ssm_state[:, None, ...].to(
            device=states.device
        )

      states = torch.cat([previous_states, states], dim=1)
      decay_chunk = torch.exp(
          segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0)))
      )
      decay_chunk = decay_chunk.transpose(1, 3)
      new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(
          dim=1
      )
      states, ssm_state = new_states[:, :-1], new_states[:, -1]

      # State to output conversion
      state_decay_out = torch.exp(A_cumsum)
      C_times_states = C[..., None, :] * states[:, :, None, ...]
      Y_off = (
          C_times_states.sum(-1)
          * state_decay_out.permute(0, 2, 3, 1)[..., None]
      )

      y = Y_diag + Y_off + D_residual
      y = y.reshape(
          batch_size, -1, self.mamba_config.n_heads, self.mamba_config.d_head
      )

      if pad_size > 0:
        y = y[:, :seq_len, :, :]
      y = y.reshape(batch_size, seq_len, -1)

      # Update cache with final state
      if mamba_cache is not None:
        mamba_cache = mamba_utils.update(mamba_cache, new_conv_state, ssm_state)

    # Apply gating and output projection
    y = y * silu(gate)
    y = self.norm(y)
    contextualized_states = self.out_proj(y.to(dtype))

    return (
        contextualized_states
        if mamba_cache is None
        else (contextualized_states, mamba_cache)
    )
