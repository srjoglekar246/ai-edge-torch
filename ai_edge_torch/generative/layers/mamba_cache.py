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

"""Utility functions for externalized Mamba Cache."""

import dataclasses
from typing import Any, List, Tuple

import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils
from ai_edge_torch.generative.layers import model_config
from ai_edge_torch.generative.utilities import types
import torch
import torch.utils._pytree as pytree


@dataclasses.dataclass
class MambaCacheEntry:
  """A single cache entry that includes conv & SSM states in MambaV2 impl."""

  conv_state: torch.Tensor
  ssm_state: torch.Tensor

  @classmethod
  def from_model_config(
      cls,
      config: model_config.MambaConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device | None = None,
      batch_size: int = 1,
  ) -> "MambaCacheEntry":
    """Build an instance of the class based on a MambaConfig."""
    conv_state_shape = (
        batch_size,
        (
            config.expand * config.hidden_size
            + 2 * config.n_groups * config.d_state
        ),
        config.d_conv,
    )
    ssm_state_shape = (
        batch_size,
        config.n_heads,
        config.d_head,
        config.d_state,
    )

    conv_state = torch.zeros(conv_state_shape, dtype=dtype, device=device)
    ssm_state = torch.zeros(ssm_state_shape, dtype=dtype, device=device)
    obj = cls(conv_state=conv_state, ssm_state=ssm_state)
    return obj


@dataclasses.dataclass
class MambaCache:
  """A utility class for holding Mamba cache entries per layer."""

  caches: Tuple[MambaCacheEntry, ...]

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device | None = None,
      batch_size: int = 1,
  ) -> "MambaCache":
    """Build an instance of the class based on model config.

    Args:
        config (ModelConfig): Model config used for building the cache.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.
        batch_size (int, optional): The batch size of the cache tensors.
          Defaults to 1.

    Returns:
        MambaCache: The created cache object.
    """
    if isinstance(config.block_configs, list):
      # Count number of blocks with non-None mamba_config
      layer_indices = [
          idx
          for idx, block_cfg in enumerate(config.block_configs)
          if block_cfg.mamba_config is not None
      ]
    elif isinstance(config.block_configs, model_config.TransformerBlockConfig):
      if config.block_configs.mamba_config is None:
        return None
      layer_indices = list(range(config.num_layers))

    caches = [
        MambaCacheEntry.from_model_config(
            config.block_config(idx).mamba_config,
            dtype,
            device,
            batch_size,
        )
        for idx in layer_indices
    ]
    obj = cls(caches=tuple(caches))
    return obj

  def flatten(self) -> List[torch.Tensor]:
    """Flatten the cache entries into a list of tensors with order conv_i, ssm_i."""
    flattened, _ = _flatten_mamba_cache(self)
    return flattened


def _flatten_mamba_cache(cache: MambaCache) -> Tuple[List[str], List[str]]:
  flattened = []
  flat_names = []
  for i, mamba_entry in enumerate(cache.caches):
    flattened.append(mamba_entry.conv_state)
    flat_names.append(f"conv_state_{i}")
    flattened.append(mamba_entry.ssm_state)
    flat_names.append(f"ssm_state_{i}")
  return flattened, flat_names


def _flatten_mamba_cache_with_keys(cache: MambaCache) -> Tuple[List, List]:
  flattened, flat_names = _flatten_mamba_cache(cache)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


def _unflatten_mamba_cache(
    values: List[torch.Tensor],
    context: Tuple[List, List],
) -> MambaCache:
  assert (
      len(values) % 2 == 0
  ), "Found odd number of conv_state and ssm_state entries."
  num_layers = len(values) // 2
  flat_names = context
  cache_entries = []
  for i in range(num_layers):
    conv_state_cache_idx = flat_names.index(f"conv_state_{i}")
    ssm_state_cache_idx = flat_names.index(f"ssm_state_{i}")
    cache_entries.append(
        MambaCacheEntry(
            conv_state=values[conv_state_cache_idx],
            ssm_state=values[ssm_state_cache_idx],
        )
    )
  obj = MambaCache(tuple(cache_entries))
  return obj


def _flatten_mamba_cache_entry(
    cache_e: MambaCacheEntry,
) -> Tuple[List[torch.Tensor], Any]:
  return (
      [cache_e.conv_state, cache_e.ssm_state],
      [],
  )


def _unflatten_mamba_cache_entry(
    values: List[torch.Tensor],
    context: Any,
) -> MambaCacheEntry:
  return MambaCacheEntry(*values)


pytree.register_pytree_node(
    MambaCacheEntry,
    _flatten_mamba_cache_entry,
    _unflatten_mamba_cache_entry,
    serialized_type_name="",
)

pytree.register_pytree_node(
    MambaCache,
    _flatten_mamba_cache,
    _unflatten_mamba_cache,
    flatten_with_keys_fn=_flatten_mamba_cache_with_keys,
    serialized_type_name="",
)


def update(
    cache: MambaCacheEntry,
    new_conv_state: torch.Tensor,
    new_ssm_state: torch.Tensor,
    use_dus: bool = True,
) -> MambaCacheEntry:
  """Out of place update of Mamba Cache buffer.

  Args:
      cache (MambaCacheEntry): The original cache buffer.
      new_conv_state (torch.Tensor): The new conv state to be updated in the cache.
      new_ssm_state (torch.Tensor): The new SSM state to be updated in the cache.

  Returns:
      MambaCacheEntry: The updated MambaCache entry based on the passed inputs.
  """
  update_mamba_cache = _update_impl if use_dus else _update_base_impl
  return update_mamba_cache(cache, new_conv_state, new_ssm_state)


def _update_base_impl(
    cache: MambaCacheEntry,
    new_conv_state: torch.Tensor,
    new_ssm_state: torch.Tensor,
) -> MambaCacheEntry:
  """Update the cache buffer without High Level Function Boundary annotation."""
  # We just replace the conv_state and ssm_state tensors, so input_pos is 0 at dim 0.
  conv_state = cache.conv_state.index_copy(
      0, torch.zeros([]).to(torch.long), new_conv_state
  )
  ssm_state = cache.ssm_state.index_copy(
      0, torch.zeros([]).to(torch.long), new_ssm_state
  )
  updated_cache = MambaCacheEntry(conv_state, ssm_state)
  return updated_cache


def _get_slice_indices(num_dims: int) -> torch.Tensor:
  """Dynamic Update Slice updates are a variadic sequence of 0-rank tensors."""
  return [torch.zeros([], dtype=torch.long) for _ in range(num_dims)]


def _update_impl(
    cache: MambaCacheEntry,
    new_conv_state: torch.Tensor,
    new_ssm_state: torch.Tensor,
) -> MambaCacheEntry:
  """Update the cache buffer for conv_state and ssm_state."""
  # NB: Here assume that input_pos == range(input_pos[0], len(input_pos))

  conv_state_indices = _get_slice_indices(3)
  ssm_state_indices = _get_slice_indices(4)

  conv_state = dus_utils.dynamic_update_slice(
      cache.conv_state, new_conv_state, conv_state_indices
  )
  ssm_state = dus_utils.dynamic_update_slice(
      cache.ssm_state, new_ssm_state, ssm_state_indices
  )

  updated_cache = MambaCacheEntry(conv_state, ssm_state)
  return updated_cache
