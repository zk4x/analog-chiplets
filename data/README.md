# Model weights

The weights are not included in this repository. The comparison scripts expect these files in this directory:

- `kws_micronet_s.safetensors` — MicroNet-S keyword-spotting weights
- `mobilenet_v1_1.0_224.safetensors` — MobileNetV1 weights
- `qwen3_linear_attention_block.safetensors` — original Qwen3 linear-attention block
- `qwen3_linear_attention_block_finetuned.safetensors` — my fine-tuned Qwen block

The Qwen fine-tune is part of this work, but the weights are intentionally not distributed here. Obtain the base model from its original source and place the fine-tuned file here if you have it.
