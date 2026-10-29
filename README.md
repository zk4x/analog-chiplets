# analog-chiplets

Independent research project: simulate memristor-based analog inference and determine which reusable analog chiplets can be built across different neural-network models.

The project maps three model workloads onto a shared set of analog components:

- **MicroNet-S** — keyword spotting, Speech Commands, 12 labels
- **MobileNetV1** — 224 px, ImageNet-1000
- **Qwen3.8-27B** — one linear-attention block, including a fine-tuned variant

The goal is not to simulate a full chip. It is to answer a design question: **which analog chiplets are reusable across models, and what does each model cost in transistors, memristors, converters, and wiring?**

## Approach

The code is organized as three layers:

- `components/` — analog circuits: crossbars, column arrays, capacitor arrays, ADCs/DACs, delta-sigma converters, translinear circuits, multipliers, dividers, square-root circuits, limiters, and support blocks.
- `kernels/` — analog chains built from components: convolution, depthwise convolution, pooling, RMSNorm, SiLU, gated delta-network state, gating, softmax, and element-wise multiply.
- `models/` — model-level chiplet chains with the ideal PyTorch reference used as a test bench.

The analog/digital boundary sits at the chiplet border. For MicroNet-S and MobileNetV1 it coincides with the model border; for the Qwen block the chiplet partition is at layer borders. The current Qwen comparison applies a converter pair at the block I/O so the block can be evaluated in one pass.

## Error model

The simulation models:

- conductance quantization
- device-to-device mismatch
- cycle/read noise
- power-law conductance drift
- read averaging
- ADC/DAC resolution and range
- delta-sigma equivalent noise

These are approximate, documented circuit-level error models, not SPICE or layout simulation. Every component carries a header table describing the parameter, the circuit element, and the error class it represents.

## Results

From the model comparison scripts:

| Model | Noiseless rel. L2 | With noise rel. L2 | KL divergence |
|---|---:|---:|---:|
| MicroNet-S | 0.0034 | 0.6715 ± 0.2286 | 0.00982 ± 0.00563 |
| MobileNetV1 | 0.0112 | 1.1459 ± 0.2205 | 0.27767 ± 0.07489 |

Qwen3.8-27B linear-attention block, original vs. fine-tuned weights:

| Weights | Ideal vs. golden | Noiseless | Default chip config | Correlation |
|---|---:|---:|---:|---:|
| Original | 0.0005 | 0.0355 | 1.2213 ± 0.5090 | 0.4561 ± 0.2231 |
| Fine-tuned | 0.2952 | 0.2981 | 0.9078 ± 0.3368 | 0.6276 ± 0.1496 |

The fine-tuned Qwen block is my own fine-tune; the weights are not included in this repository.

The full sizing and BOM data is in [`docs/chiplet_sizing.xlsx`](docs/chiplet_sizing.xlsx):

- `config` — tile dimensions, cell bits, spare fraction, transistor budgets per block
- `micronet-s` — per-layer sizing, transistor/memristor counts, and error results
- `mobilenetv1` — per-layer sizing and error results
- `qwen_block` — Qwen block sizing, chiplets, converters, and error results

## Repository layout

```text
components/     analog circuit models
kernels/        analog kernel chains
models/         model-level comparisons
docs/           sizing/BOM spreadsheet
data/           model weights (not included)
```

## Running

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the model weights in `data/` (see `data/README.md`), then run a comparison:

```sh
python models/micronet_s_comparison.py
python models/mobilenet_v1_comparison.py
python models/qwen_comparison.py
```

## Data and weights

Model weights are **not** included: neither the public model weights nor the fine-tuned Qwen block. They are large and have their own licenses. Put them in `data/` to reproduce the comparisons.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
