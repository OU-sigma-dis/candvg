# CandVG
This repository contains the code for the experiments presented in the following paper:

**Recognizing Dialogue Ambiguity via Candidate Set Visual Grounding**, Shun Inadumi, Lee Sangmyeong and Koichiro Yoshino.

<!-- [![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)]() -->
[![Hugging Face](https://img.shields.io/badge/HuggingFace-green)](https://huggingface.co/datasets/SInadumi/CandVG)

## Setup

1. Follow the instructions in [data/README.md](data/README.md) to set up the datasets.

2. Install the dependencies:
```bash
uv sync --group normal
```

3. Optionally, authenticate with Hugging Face and prepare the default model:
```bash
uv run hf auth login
uv run python scripts/download_models.py --model-id Qwen/Qwen3.5-2B
```

The checkpoint is saved to `checkpoints/Qwen/Qwen3.5-2B`.

## Run inference
Run VLM inference as follows:
```bash
uv run python scripts/run_task.py \
  --model checkpoints/Qwen/Qwen3.5-2B \
  --dataset candvg \
  --target-mode single \
  --max-samples 10
```
`single` corresponds to the CVG setting, while `candidates` corresponds to the CSVG setting.

Inference artifacts are written to:
```text
outputs/candvg_single/Qwen3.5-2B/[timestamp]/
├── inference_outputs.jsonl
├── predictions/
└── run_config.json
```

## Evaluate predictions

Evaluate the `predictions/` produced by `run_task.py`:

```bash
uv run python scripts/eval_predictions.py \
  --dataset candvg \
  --target-mode single \
  --exp-dir outputs/candvg_single/Qwen3.5-2B/[timestamp]
```

The evaluation writes `metrics.json` and `evaluation_outputs.jsonl` to the
same experiment directory.

Optionally, visualize the evaluation outputs:
```bash
uv run python scripts/misc/visualize_outputs.py \
  --exp-dir outputs/candvg_single/Qwen3.5-2B/[timestamp]
```

<!-- 
## Citation
Please cite our paper:

```bibtex
@inproceedings{Inadumi-etal_2026_CandVG,
  author    = {Inadumi, Shun and Sangmyeong, Lee and Yoshino, Koichiro},
  title     = {Recognizing Dialogue Ambiguity via Candidate Set Visual Grounding},
  booktitle = {TODO},
  year      = {2026},
  pages = {TODO},
}
```

Please also cite InViG:

```bibtex
@inproceedings{Zhang-etal_2024_InViG,
    author    = {Zhang, Hanbo and Xu, Jie and Mo, Yuchen and Kong, Tao},
    title     = {{InViG}: Benchmarking Open-Ended Interactive Visual Grounding with 500K Dialogues},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2024},
    pages     = {5508--5518},
}
```

and Open Images:

```bibtex
@article{Kuznetsova-etal_2020_OpenImages,
  author  = {Kuznetsova, Alina and Rom, Hassan and Alldrin, Neil and Uijlings, Jasper and Krasin, Ivan and Pont-Tuset, Jordi and Kamali, Shahab and Popov, Stefan and Malloci, Matteo and Kolesnikov, Alexander and Duerig, Tom and Ferrari, Vittorio},
  title   = {{The Open Images Dataset V4}: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale},
  journal = {International Journal of Computer Vision (IJCV)},
  year    = {2020},
  volume  = {128},
  number  = {7},
  pages   = {1956--1981},
  doi     = {10.1007/s11263-020-01316-z},
  url     = {https://doi.org/10.1007/s11263-020-01316-z},
}
```