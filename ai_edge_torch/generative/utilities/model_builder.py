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

"""Utilities to be used for re-authoring transformer models."""

import copy
from typing import Callable, Dict
from typing import Optional, Tuple

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import mamba_cache as mamba_utils
from ai_edge_torch.generative.layers import lora as lora_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import export_config as export_cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
)

TENSOR_NAMES_WITH_SEPARATE_LM_HEAD = copy.copy(TENSOR_NAMES)
TENSOR_NAMES_WITH_SEPARATE_LM_HEAD.lm_head = "lm_head"


class DecoderOnlyModel(nn.Module):
  """A simple decoder-only transformer model built from the Edge Generative API.

  This model is used for re-authoring. model_config is used to specify the
  details of model architecture and parameters.

  It assumes that the attention configs for ROPE, i.e. head_dim, rotary_base,
  and rotary_percentage are the same for all layers.
  """

  def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=config.embedding_padding_idx
    )
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    if config.lm_head_share_weight_with_embedding:
      self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim, config.final_norm_config
    )
    self.config = config
    self.build_mask_cache(mask_cache_size)

  def build_mask_cache(self, mask_cache_size: int):
    assert (
        mask_cache_size <= self.config.max_seq_len
    ), "Mask cache size must be less than or equal to the max seq length."
    if mask_cache_size <= 0:
      self.mask_cache = None
    else:
      self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size)
    
  def first_attn_block(self) -> Optional[attention.TransformerBlock]:
    """Returns the first attention block."""
    for block in self.transformer_blocks:
      if block.atten_func is not None:
        return block
    return None
  
  def num_atten_blocks(self) -> int:
    """Returns the number of attention blocks."""
    count = 0
    for block in self.transformer_blocks:
      if block.atten_func is not None:
        count += 1
    return count
  
  def has_mamba_blocks(self) -> bool:
    """Returns whether the model has any Mamba blocks."""
    for block in self.transformer_blocks:
      if block.mamba_func is not None:
        return True
    return False

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache = None,
      mask: Optional[torch.Tensor] = None,
      lora: Optional[lora_utils.LoRA] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
      mamba_cache: mamba_utils.MambaCache = None,
      mamba_mask: Optional[torch.Tensor] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    # token embeddings of shape (b, t, n_embd)
    input_embeds = self.tok_embedding(tokens)

    # ROPE parameters for all attn_configs are the same. Take the first one.
    first_attn_block = self.first_attn_block()
    n_elem = 0
    rope = None
    if first_attn_block is not None:
      attn_config = first_attn_block.config.attn_config
      n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
      rope = self.config.build_rope(input_pos, n_elem, attn_config.rotary_base)

    if mask is None:
      assert self.mask_cache is not None, "Mask cache must be built."
      assert kv_cache is not None, "KV cache must be provided."
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, :kv_cache.get_max_seq_len()]

    return self._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, lora, export_config, mamba_cache, mamba_mask
    )

  def _forward_with_embeds(
      self,
      input_embeds: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
      mask: torch.Tensor,
      input_pos: Optional[torch.Tensor],
      kv_cache: kv_utils.KVCache = None,
      lora: Optional[lora_utils.LoRA] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
      mamba_cache: mamba_utils.MambaCache = None,
      mamba_mask: Optional[torch.Tensor] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    """Forwards the model with input embeddings."""
    assert self.num_atten_blocks() == len(kv_cache.caches), (
        "The number of transformer attention blocks and the number of KV cache entries"
        " must be the same."
    )

    x = input_embeds
    if self.config.embedding_scale is not None:
      x = x * self.config.embedding_scale

    # TODO(srjoglekar): Combine kv_cache and mamba_cache into a single cache?
    updated_kv_entries = []
    updated_mamba_entries = []
    kv_layer_idx = 0
    mamba_layer_idx = 0
    updated_kv_cache = None
    updated_mamba_cache = None
    for i, block in enumerate(self.transformer_blocks):
      if block.atten_func is not None:
        kv_entry = kv_cache.caches[kv_layer_idx] if kv_cache else None
        lora_adapter = lora.adapters[i] if lora else None
        x, kv_entry = block(x, rope, mask, input_pos, kv_entry, lora_adapter)
        if kv_entry:
          updated_kv_entries.append(kv_entry)
        kv_layer_idx += 1
      elif block.mamba_func is not None:
        mamba_entry = mamba_cache.caches[mamba_layer_idx] if mamba_cache else None
        lora_adapter = lora.adapters[i] if lora else None
        x, mamba_entry = block(
            x, rope=None, mask=mask, input_pos=input_pos, kv_cache=None,
            lora=lora_adapter, mamba_cache=mamba_entry, mamba_mask=mamba_mask
        )
        if mamba_entry:
          updated_mamba_entries.append(mamba_entry)
        mamba_layer_idx += 1
    cache_out_dir = {}
    if len(updated_kv_entries) > 0:
      updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))
      cache_out_dir["kv_cache"] = updated_kv_cache
    if len(updated_mamba_entries) > 0:
      updated_mamba_cache = mamba_utils.MambaCache(tuple(updated_mamba_entries))
      cache_out_dir["mamba_cache"] = updated_mamba_cache

    if export_config is not None:
      if (
          torch.numel(input_pos) > 1
          and not export_config.output_logits_on_prefill
      ):
        return cache_out_dir

    x = self.final_norm(x)
    logits = self.lm_head(x)  # (b, t, vocab_size)
    if self.config.final_logit_scaling:
      logits = logits / self.config.final_logit_scaling
    return {"logits": logits, **cache_out_dir}


def build_decoder_only_model(
    checkpoint_path: str,
    config: cfg.ModelConfig,
    tensor_names: loading_utils.ModelLoader.TensorNames,
    model_class: type[nn.Module] = DecoderOnlyModel,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  transformer = model_class(config, mask_cache_size)
  loader = loading_utils.ModelLoader(
      checkpoint_path, tensor_names, custom_loader
  )
  loader.load(
      transformer, strict=not config.lm_head_share_weight_with_embedding
  )
  transformer.eval()
  return transformer
