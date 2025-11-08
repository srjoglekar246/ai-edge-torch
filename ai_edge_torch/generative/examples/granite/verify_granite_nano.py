"""Verifies the reauthored Granite 4 Nano models."""

import logging
import os
import pathlib
from typing import Callable, Dict

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.granite.granite_nano import build_nano4_350m_model
from ai_edge_torch.generative.examples.granite.granite_nano import build_nano4h_350m_model
from ai_edge_torch.generative.examples.granite.granite_nano import granite_nano_custom_loader
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import torch
import transformers

_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)
_MODEL_SIZE = flags.DEFINE_string(
    "model_size",
    "350m",
    'Size of the granite model. Currently only supports "350m".',
)
_IS_HYBRID = flags.DEFINE_bool(
    "is_hybrid",
    False,
    "Whether to use the hybrid version of the granite model.",
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir",
    "",
    "Path to the checkpoint directory containing the model files.",
)


def verify_granite_nano(
    checkpoint_dir: str,
    weight_filename: str = "model.safetensors",
    max_new_tokens: int = 30,
    prompts: list[str] | None = None,
    initialize_from_local: bool = True,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
    is_hybrid: bool = False,
) -> bool:
  """Verifies the reauthored Granite Nano model with a custom loader."""
  print("Loading the original model from: %s", checkpoint_dir)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(
      checkpoint_dir
  )

  print("Building the reauthored model from: %s", checkpoint_dir)
  if custom_loader is None and not initialize_from_local:
    custom_loader = loader.get_custom_loader("", "safetensors")

  if initialize_from_local:
    # Locate the cached dir.
    cached_config_file = transformers.utils.cached_file(
        checkpoint_dir, transformers.utils.CONFIG_NAME
    )
    reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  else:
    reauthored_checkpoint = os.path.join(checkpoint_dir, weight_filename)

  print("Building the reauthored model from: %s", reauthored_checkpoint)

  # Select model builder based on is_hybrid flag
  if is_hybrid:
    model_builder = build_nano4h_350m_model
  else:
    model_builder = build_nano4_350m_model

  reauthored_model = model_builder(
      checkpoint_path=reauthored_checkpoint,
      custom_loader=custom_loader,
      mask_cache_size=verifier.DEFAULT_KV_CACHE_MAX_LEN,
  )

  print("Loading the tokenizer from: %s", checkpoint_dir)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
  return verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=_PROMPTS if prompts is None else prompts,
      max_new_tokens=max_new_tokens,
      atol=1e-04,
  )


def main(_):
  if not _CHECKPOINT_DIR.value:
    raise ValueError("checkpoint_dir flag must be provided.")

  if _MODEL_SIZE.value != "350m":
    raise ValueError(
        f"Unsupported model_size: {_MODEL_SIZE.value}. Only '350m' is"
        " supported."
    )

  verify_granite_nano(
      checkpoint_dir=_CHECKPOINT_DIR.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      prompts=_PROMPTS.value,
      custom_loader=granite_nano_custom_loader,
      is_hybrid=_IS_HYBRID.value,
  )


if __name__ == "__main__":
  app.run(main)
