import argparse
from collections.abc import Sequence
import sys
import time

from ai_edge_litert import interpreter as interpreter_lib
import numpy as np
from transformers import AutoTokenizer


class LiteRTLlmPipeline:

  def __init__(
      self, interpreter, tokenizer, top_p=1.0, top_k=0, temperature=1.0
  ):
    """Initializes the pipeline."""
    self._interpreter = interpreter
    self._tokenizer = tokenizer
    self._top_p = top_p
    self._top_k = top_k
    self._temperature = temperature

    self._prefill_runner = None
    self._decode_runner = self._interpreter.get_signature_runner("decode")

    self._needs_mamba_mask = False

  def _init_prefill_runner(self, num_input_tokens: int):
    """Initializes all the variables related to the prefill runner.

    This method initializes the following variables:
      - self._prefill_runner: The prefill runner based on the input size.
      - self._max_seq_len: The maximum sequence length supported by the model.
      - self._max_kv_cache_seq_len: The maximum sequence length supported by the
        KV cache.

    Args:
      num_input_tokens: The number of input tokens.
    """
    if not self._interpreter:
      raise ValueError("Interpreter is not initialized.")

    # Prefill runner related variables will be initialized in `predict_text` and
    # `compute_log_likelihood`.
    self._prefill_runner = self._get_prefill_runner(num_input_tokens)
    # input_token_shape has shape (batch, max_seq_len)
    input_token_shape = self._prefill_runner.get_input_details()["tokens"][
        "shape"
    ]
    if len(input_token_shape) == 1:
      self._max_seq_len = input_token_shape[0]
    else:
      self._max_seq_len = input_token_shape[1]

    # kv cache input has shape [batch=1, seq_len, num_heads, dim].
    kv_cache_shape = self._prefill_runner.get_input_details()["kv_cache_k_0"][
        "shape"
    ]
    self._max_kv_cache_seq_len = kv_cache_shape[1]

  def _init_caches(self) -> dict[str, np.ndarray]:
    if self._prefill_runner is None:
      raise ValueError("Prefill runner is not initialized.")
    caches = {}
    for input_key in self._prefill_runner.get_input_details().keys():
      if "kv_cache" in input_key or "mamba_cache" in input_key:
        caches[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
    return caches

  def _get_prefill_runner(self, num_input_tokens: int):
    """Gets the prefill runner with the best suitable input size.

    Args:
      num_input_tokens: The number of input tokens.

    Returns:
      The prefill runner with the smallest input size.
    """
    best_signature = None
    delta = sys.maxsize
    max_prefill_len = -1
    for key in self._interpreter.get_signature_list().keys():
      if "prefill" not in key:
        continue
      input_pos = self._interpreter.get_signature_runner(
          key
      ).get_input_details()["input_pos"]
      self._needs_mamba_mask = "mamba_mask" in self._interpreter.get_signature_runner(key).get_input_details()
      # input_pos["shape"] has shape (max_seq_len, )
      seq_size = input_pos["shape"][0]
      max_prefill_len = max(max_prefill_len, seq_size)
      if num_input_tokens <= seq_size and seq_size - num_input_tokens < delta:
        delta = seq_size - num_input_tokens
        best_signature = key
    if best_signature is None:
      raise ValueError(
          "The largest prefill length supported is %d, but we have %d number of"
          " input tokens" % (max_prefill_len, num_input_tokens)
      )
    return self._interpreter.get_signature_runner(best_signature)

  def _run_prefill(
      self,
      prefill_token_ids: Sequence[int],
  ) -> dict[str, np.ndarray]:
    """Runs prefill and returns the kv cache.

    Args:
      prefill_token_ids: The token ids of the prefill input.

    Returns:
      The updated kv cache.
    """
    if not self._prefill_runner:
      raise ValueError("Prefill runner is not initialized.")
    prefill_token_length = len(prefill_token_ids)
    if prefill_token_length == 0:
      return self._init_caches()

    # Prepare the input to be [1, max_seq_len].
    input_token_ids = [0] * self._max_seq_len
    input_token_ids[:prefill_token_length] = prefill_token_ids
    input_token_ids = np.asarray(input_token_ids, dtype=np.int32)
    input_token_ids = np.expand_dims(input_token_ids, axis=0)

    # Prepare the input position to be [max_seq_len].
    input_pos = [0] * self._max_seq_len
    input_pos[:prefill_token_length] = range(prefill_token_length)
    input_pos = np.asarray(input_pos, dtype=np.int32)

    # Initialize kv cache.
    prefill_inputs = self._init_caches()
    prefill_inputs.update({
        "tokens": input_token_ids,
        "input_pos": input_pos,
    })
    if self._needs_mamba_mask:
      mamba_mask = np.zeros((1, self._max_seq_len), dtype=np.float32)
      mamba_mask[0, :prefill_token_length] = 1.0
      prefill_inputs["mamba_mask"] = mamba_mask
    prefill_outputs = self._prefill_runner(**prefill_inputs)
    if "logits" in prefill_outputs:
      # Prefill outputs includes logits and kv cache. We only output kv cache.
      prefill_outputs.pop("logits")

    return prefill_outputs

  def _greedy_sampler(self, logits: np.ndarray) -> int:
    return int(np.argmax(logits))

  def _sample_token(self, logits: np.ndarray) -> int:
    """Sample token using top-p, top-k, and temperature."""
    if self._temperature == 0.0 or (self._top_p == 1.0 and self._top_k == 0):
      return self._greedy_sampler(logits)

    # Apply temperature
    if self._temperature != 1.0:
      logits = logits / self._temperature

    # Apply top-k filtering
    if self._top_k > 0:
      top_k_indices = np.argpartition(logits, -self._top_k)[-self._top_k :]
      top_k_logits = np.full_like(logits, -np.inf)
      top_k_logits[top_k_indices] = logits[top_k_indices]
      logits = top_k_logits

    # Convert to probabilities
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)

    # Apply top-p filtering
    if self._top_p < 1.0:
      sorted_indices = np.argsort(probs)[::-1]
      sorted_probs = probs[sorted_indices]
      cumsum_probs = np.cumsum(sorted_probs)
      cutoff_idx = np.searchsorted(cumsum_probs, self._top_p) + 1

      top_p_indices = sorted_indices[:cutoff_idx]
      filtered_probs = np.zeros_like(probs)
      filtered_probs[top_p_indices] = probs[top_p_indices]
      probs = filtered_probs / np.sum(filtered_probs)

    # Sample from the distribution
    return np.random.choice(len(probs), p=probs)

  def _run_decode(
      self,
      start_pos: int,
      start_token_id: int,
      cache: dict[str, np.ndarray],
      max_decode_steps: int,
  ) -> tuple[str, float, float]:
    """Runs decode and outputs the token ids from sampler.

    Args:
      start_pos: The position of the first token of the decode input.
      start_token_id: The token id of the first token of the decode input.
      cache: The kv/mamba cache from the prefill.
      max_decode_steps: The max decode steps.

    Returns:
      A tuple of (decoded_text, ttft_ms, avg_tbt_ms).
    """
    next_pos = start_pos
    next_token = start_token_id
    decode_text = []
    decode_inputs = cache

    ttft_ms = None
    token_times = []

    for step in range(max_decode_steps):
      step_start_time = time.time()

      decode_inputs.update({
          "tokens": np.array([[next_token]], dtype=np.int32),
          "input_pos": np.array([next_pos], dtype=np.int32),
      })
      if self._needs_mamba_mask:
        mamba_mask = np.zeros((1, 1), dtype=np.float32)
        mamba_mask[0, 0] = 1.0
        decode_inputs["mamba_mask"] = mamba_mask
      decode_outputs = self._decode_runner(**decode_inputs)

      step_end_time = time.time()
      step_time_ms = (step_end_time - step_start_time) * 1000

      if step == 0:
        ttft_ms = step_time_ms
      else:
        token_times.append(step_time_ms)

      # Output logits has shape (batch=1, 1, vocab_size). We only take the first
      # element.
      logits = decode_outputs.pop("logits")[0][0]
      next_token = self._sample_token(logits)
      if next_token == self._tokenizer.eos_token_id:
        break
      decode_text.append(
          self._tokenizer.decode(next_token, skip_special_tokens=False)
      )
      print(decode_text[-1], end="", flush=True)
      # Decode outputs includes logits and kv cache. We already poped out
      # logits, so the rest is kv cache. We pass the updated kv cache as input
      # to the next decode step.
      decode_inputs = decode_outputs
      next_pos += 1

    print()  # print a new line at the end.
    avg_tbt_ms = np.mean(token_times) if token_times else 0.0
    return "".join(decode_text), ttft_ms or 0.0, avg_tbt_ms

  def generate(self, prompt: str, max_decode_steps: int | None = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    token_ids = self._tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    # Initialize the prefill runner with the suitable input size.
    self._init_prefill_runner(len(token_ids))

    # Run prefill.
    # Prefill up to the seond to the last token of the prompt, because the last
    # token of the prompt will be used to bootstrap decode.
    prefill_token_length = len(token_ids) - 1

    print("Running prefill")
    cache = self._run_prefill(token_ids[:prefill_token_length])
    # Run decode.
    print("Running decode")
    actual_max_decode_steps = (
        self._max_kv_cache_seq_len - prefill_token_length - 1
    )
    if max_decode_steps is not None:
      actual_max_decode_steps = min(actual_max_decode_steps, max_decode_steps)
    decode_text, ttft_ms, avg_tbt_ms = self._run_decode(
        prefill_token_length,
        token_ids[prefill_token_length],
        cache,
        actual_max_decode_steps,
    )

    print(f"\nMetrics:")
    print(f"Time to First Token (TTFT): {ttft_ms:.2f} ms")
    print(f"Average Time Between Tokens (TBT): {avg_tbt_ms:.2f} ms")

    return decode_text


def main():
  parser = argparse.ArgumentParser(
      description="Run TFLite inference with Granite model"
  )
  parser.add_argument(
      "--model_path",
      type=str,
      required=True,
      help="Path to the TFLite model file",
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      required=True,
      help="Path to the tokenizer directory",
  )
  parser.add_argument(
      "--prompt",
      type=str,
      default="What is the best place to visit in New York?",
      help="Input prompt for generation",
  )
  parser.add_argument(
      "--top_p",
      type=float,
      default=1.0,
      help="Top-p sampling parameter (default: 1.0)",
  )
  parser.add_argument(
      "--top_k",
      type=int,
      default=0,
      help="Top-k sampling parameter (default: 0)",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=1.0,
      help="Temperature for sampling (default: 1.0)",
  )
  parser.add_argument(
      "--max_decode_steps",
      type=int,
      default=None,
      help="Maximum number of decode steps",
  )

  args = parser.parse_args()

  interpreter = interpreter_lib.InterpreterWithCustomOps(
      custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
      model_path=args.model_path,
      num_threads=2,
      experimental_default_delegate_latest_features=True,
  )
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  # Disclaimer: Model performance demonstrated with the Python API in this notebook is not representative of performance on a local device.
  pipeline = LiteRTLlmPipeline(
      interpreter,
      tokenizer,
      top_p=args.top_p,
      top_k=args.top_k,
      temperature=args.temperature,
  )

  pipeline.generate(args.prompt, max_decode_steps=args.max_decode_steps)


if __name__ == "__main__":
  main()
