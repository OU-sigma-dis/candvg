# Setup Datasets
## Download Data
Download the annotations and images from the sources listed below and place them under data/ following this directory structure.

```text
data/
├── candvg/
│   ├── candvg_original.jsonl
│   ├── bin_counts.json
│   └── images/
├── flickr30k_entities/
│   ├── final_flickr_separateGT_val.json
│   ├── final_flickr_separateGT_test.json
│   └── flickr30k_images/
├── coco/
│   ├── train2014/
│   ├── refcoco/
│   │   └── refcoco_{val,testA,testB}.jsonl
│   ├── refcoco+/
│   │   └── refcoco+_{val,testA,testB}.jsonl
│   └── refcocog/
│       └── refcocog_{val,test}.jsonl
└── gRefCOCO/
    ├── grefs(unc).json
    └── instances.json
```

CandVG images are loaded from  `data/candvg/images/`. Flickr30k Entities images
are loaded from  the `flickr30k_images/` directory beside each annotation file.
gRefCOCO uses COCO images from `data/coco/train2014/`. RefCOCO, RefCOCO+, and
RefCOCOg JSONL records contain their image paths in the `image` field, so those
paths must point to the corresponding COCO images.

### Dataset Sources
- [CandVG](https://huggingface.co)
- [Flickr30k Entities](https://github.com/bryanplummer/flickr30k_entities)
- [COCO](https://cocodataset.org/#home)
- [RefCOCO/RefCOCO+/RefCOCOg](https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/EVALUATION.md)
- [gRefCOCO](https://github.com/henghuiding/gRefCOCO)


## Prepare CandVG Benchmark

The distributed `data/candvg/candvg_original.jsonl` excludes data derived from
InViG. Reconstruct the complete benchmark locally using the original InViG
annotations before running CandVG experiments. The reconstruction preserves
`candvg_original.jsonl` and writes the completed annotations to
`data/candvg/candvg.jsonl`.

1. Download the original InViG validation and test annotations.

```bash
wget https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/invig_icra2024/invig21k_anns/raw/invig21k_valid_anns.jsonl
wget https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/invig_icra2024/invig21k_anns/raw/invig21k_test_anns.jsonl
```

2. Restore the InViG-derived data.

```bash
uv run python scripts/misc/setup_candvg.py \
    --valid-path ./invig21k_valid_anns.jsonl \
    --test-path ./invig21k_test_anns.jsonl
```

Use the generated `data/candvg/candvg.jsonl` as the annotation file for CandVG
inference and evaluation. 