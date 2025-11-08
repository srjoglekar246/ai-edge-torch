# Granite 4 Nano Models - AI Edge Torch Conversion

This folder implements the re-authoring and conversion of IBM Granite 4 Nano models via ai-edge-torch to LiteRT/TFLite. 

The main feature of the Granite models is that they come in two variants:
- **Dense**: Traditional transformer architecture with attention layers
- **Hybrid**: Mixture of MambaV2 and Attention layers for token mixing

## Workflow

### 1. Download Hugging Face Assets

Download the model assets to a local directory:

```bash
hf download ibm-granite/granite-4.0-h-350m --local-dir ./nano_4h_350m_assets
```

For the dense variant, use `ibm-granite/granite-4.0-350m` instead.

### 2. Verify Re-authored Model

Verify that the re-authored model matches the original:

```bash
python verify_granite_nano.py --checkpoint_dir <path_to_assets>
```

For hybrid models, add the `--is_hybrid` flag:
```bash
python verify_granite_nano.py --is_hybrid --checkpoint_dir <path_to_assets>
```

### 3. Convert Model to TFLite

Convert the model to TFLite format:

```bash
python convert_to_tflite.py --checkpoint_path <path_to_assets> --output_path <output_directory> --prefill_seq_lens 64 256
```

For hybrid models, add the `--is_hybrid` flag:
```bash
python convert_to_tflite.py --checkpoint_path <path_to_assets> --output_path <output_directory> --prefill_seq_lens 64 256 --is_hybrid
```

### 4. Run Inference

Run inference with the converted TFLite model:

```bash
python run_tflite.py --model_path <path_to_tflite_model> --tokenizer <path_to_tokenizer> --prompt "<your_prompt>"
```

The model path will be:
- Dense: `granite_q8_ekv1280.tflite`
- Hybrid: `granite_hybrid_q8_ekv1280.tflite`

Additional inference options:
- `--top_p`: Top-p sampling parameter (default: 1.0)
- `--top_k`: Top-k sampling parameter (default: 0)  
- `--temperature`: Temperature for sampling (default: 1.0)
- `--max_decode_steps`: Maximum number of decode steps

## Model Variants

Currently supported:
- **Model Size**: 350M parameters only
- **Architectures**: Dense and Hybrid variants
- **Quantization**: Q8 quantization by default
