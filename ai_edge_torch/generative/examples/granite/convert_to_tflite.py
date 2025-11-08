"""Example of converting Granite 4 Nano models to multi-signature tflite model."""

from absl import app
from absl import flags as absl_flags
from ai_edge_torch.generative.examples.granite import granite_nano
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config as export_config_lib
from ai_edge_torch.generative.utilities import loader

flags = converter.define_conversion_flags("granite")

# Add conversion flags for granite models
absl_flags.DEFINE_string(
    "model_size",
    "350m",
    'Size of the granite model. Currently only supports "350m".',
)
absl_flags.DEFINE_bool(
    "is_hybrid",
    False,
    "Whether to use the hybrid version of the granite model.",
)


def main(_):
  # Fix undefined variables
  checkpoint_path = flags.FLAGS.checkpoint_path
  output_name_prefix = flags.FLAGS.output_name_prefix

  # Append "_hybrid" to output name if using hybrid model
  if flags.FLAGS.is_hybrid:
    output_name_prefix += "_hybrid"

  # Select model builder based on flags
  if flags.FLAGS.model_size == "350m":
    if flags.FLAGS.is_hybrid:
      model_builder = granite_nano.build_nano4h_350m_model
    else:
      model_builder = granite_nano.build_nano4_350m_model
  else:
    raise ValueError(
        f"Unsupported model_size: {flags.FLAGS.model_size}. Only '350m' is"
        " supported."
    )

  pytorch_model = model_builder(
      checkpoint_path,
      granite_nano.granite_nano_custom_loader,
      converter.get_mask_cache_size_from_flags(),
  )

  # Extra model for GPU dynamic shape verification if needed.
  extra_model = None
  extra_prefill_seq_lens = None
  extra_kv_cache_max_len = 0
  if flags.FLAGS.gpu_dynamic_shapes:
    prefill_seq_lens = [
        converter.get_magic_number_for(l) for l in flags.FLAGS.prefill_seq_lens
    ]
    kv_cache_max_len = converter.get_magic_number_for(
        flags.FLAGS.kv_cache_max_len
    )

    if flags.FLAGS.export_gpu_dynamic_shape_verifications:
      extra_kv_cache_max_len = converter._CONTEXT_LENGTH_TO_VERIFY_MAGIC_NUMBERS
      if extra_kv_cache_max_len > flags.FLAGS.kv_cache_max_len:
        extra_kv_cache_max_len = flags.FLAGS.kv_cache_max_len
      extra_model = model_builder(
          checkpoint_path,
          loader.maybe_get_custom_loader(
              checkpoint_path, flags.FLAGS.custom_checkpoint_loader
          ),
          extra_kv_cache_max_len,
      )
      extra_prefill_seq_lens = []
      if (
          extra_kv_cache_max_len
          > converter._SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
      ):
        extra_prefill_seq_lens.append(
            converter._SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
        )
      if (
          extra_kv_cache_max_len
          > converter._LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
      ):
        extra_prefill_seq_lens.append(
            converter._LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
        )
  else:
    prefill_seq_lens = flags.FLAGS.prefill_seq_lens
    kv_cache_max_len = flags.FLAGS.kv_cache_max_len

  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=output_name_prefix,
      prefill_seq_len=prefill_seq_lens,
      kv_cache_max_len=kv_cache_max_len,
      quantize=flags.FLAGS.quantize,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=export_config_lib.get_from_flags(),
      extra_model=extra_model,
      extra_prefill_seq_lens=extra_prefill_seq_lens,
      extra_kv_cache_max_len=extra_kv_cache_max_len,
      extra_signature_prefix="test_" if extra_model is not None else "",
  )


if __name__ == "__main__":
  app.run(main)
