# CogniTune

Domain-specialized AI/ML tutor model fine-tuned from Qwen2.5-3B-Instruct using LoRA on Apple Silicon (M5 Pro) via MLX.

## Model
[Pickamon/CogniTune-Qwen2.5-3B](https://huggingface.co/Pickamon/CogniTune-Qwen2.5-3B) on HuggingFace

## What It Does
Standard LLMs answer AI/ML questions like encyclopedias. CogniTune answers like a tutor — leading with analogies, correcting misconceptions, compressing to a one-liner.

## Results
| Checkpoint | Val Loss |
|------------|----------|
| Iter 1 | 2.799 |
| Iter 100 | **1.616** ← best |
| Iter 200 | 1.976 |
| Iter 300 | 2.227 |

- Varied format data reduced best val loss by 11% over uniform templates
- Optimal early stopping at 100 iterations
- Style transfer confirmed — factual accuracy orthogonal to fine-tuning

## Training
- Base model: Qwen/Qwen2.5-3B-Instruct
- Method: LoRA (layers=8, rank=8, lr=5e-5)
- Framework: MLX
- Hardware: Apple M5 Pro 24GB
- Dataset: ~460 hand-crafted AI/ML Q&A pairs
