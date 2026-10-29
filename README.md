# SOCLR-Segformer

RGB crack segmentation with direction/position guided feature aggregation,
original-resolution selective refinement, direction confidence gating and fixed
multiscale probability fusion. The name matches the submitted VLP method.

## Results and scope

The user-supplied leaderboard screenshot reports mIoU 81.94%, Crack F1 79.83%,
Crack IoU 66.43%, Crack Precision 78.07%, Crack Recall 81.67%, aAcc 97.57%.
It shows position 13 among 84 methods at capture time; ranks can change.
Local development Val uses 312 original images, scored once after four-flip TTA:
macro Dice 0.7903864426. This is a different evaluation from the public leaderboard.

The final method combines full multiscale 60%, selective refinement 20%, and
direction gate 20%; threshold 0.47. Full scales 1.0/1.25/1.5 are equally averaged.
Refinement and gate use their trained 1.25x geometry. All probabilities are restored
to 672x378 before fusion. No denoising or context verification is included.

## Environment

Python 3.10, PyTorch 2.8.0, torchvision 0.23.0, SMP 0.5.0. Install an appropriate
PyTorch CUDA build first, then `pip install -r requirements.txt`.
The measured machine used an RTX 4060 8 GB, FP32.

## Weights and inference

Place full.pt, selective_refine.pt, direction_gate.pt in `weights/`.
Their exact SHA256 hashes are in `configs/final.json`; inference verifies them.
Public model downloads: https://github.com/ly884728889-dot/LIUYAN/releases/tag/v1.0.0

- [full.pt](https://github.com/ly884728889-dot/LIUYAN/releases/download/v1.0.0/full.pt)
- [selective_refine.pt](https://github.com/ly884728889-dot/LIUYAN/releases/download/v1.0.0/selective_refine.pt)
- [direction_gate.pt](https://github.com/ly884728889-dot/LIUYAN/releases/download/v1.0.0/direction_gate.pt)

All three assets were downloaded completely without authentication and matched their
local SHA256 hashes. See `verification/public_upload.json` and
`REPRODUCTION_VERIFICATION.json`. No credentials or dataset images are included.

```bash
python predict.py --images /path/to/Benchmark/img --weights-dir weights --output outputs/submission
```

Outputs: 300 native-size grayscale PNGs with pixel values 0/1, and result.zip with
exactly `result/<original_stem>.png` members. Existing output directories are refused.
For smaller local smoke tests use `--expected-count N` and a separate image folder.
CPU works with `--device cpu`, but is much slower.

## Training history and reproducibility

Train 2184 originals expanded by four synchronized flips to 8736 views; Val 312 originals.
The split is inherited from the competition study; Train/Val parent groups are disjoint.
No road/session independence is claimed. Historical Benchmark parent overlaps were
explicitly retained; Benchmark pixels and leaderboard outcomes were not used for
the local threshold/weight search in this release.

Full and selective refinement trained 50 epochs from ImageNet MiT-B2 initialization,
physical batch 2 with accumulation 2, SGD lr .01, momentum .9, weight decay 0,
cosine schedule minimum 1e-5; BCE + per-image soft Dice + .1 direction CE + .1 position Dice.
Their selected checkpoints are epochs 8 and 8. Direction gate's selected checkpoint
is original epoch 9. The interrupted gate later ran labels 10-50 from that checkpoint,
but did not improve it; do not describe it as an uninterrupted fair 50-epoch control.
Inference is RGB-only; mask-derived directions are training supervision only.

Training entrypoint and exact generator are included. New runs are independent
reproductions and are not promised to be bitwise-identical to historical CUDA runs.
CUDA grid_sample backward was observed to be nondeterministic. Historical weights
are provided for exact inference reproduction rather than new training claims.

## Licenses

Original additions are MIT. SegFormer/MiT dependencies and pretrained weights retain
their upstream NVIDIA restrictions; see THIRD_PARTY_NOTICES.md. Dataset redistribution
is excluded. No claim of a new Transformer backbone or universal structural gain is made.

## Independent training

```bash
python train.py --arm full --inventory splits/data_inventory.csv --train-list splits/train_images.txt --val-list splits/val_images.txt --data-root /path/to/original-project-data-root --encoder-path /path/to/mit_b2.pth --output outputs/full_seed42
```

Use `--arm selective_refine` or `--arm direction_gate` for the other architectures.
The released gate is epoch 9 of its original run. To reproduce the later separate
warm-start study use `--warm-start weights/direction_gate.pt --start-epoch 10 --last-epoch 50`;
this resets optimizer/RNG and is not an uninterrupted 50-epoch experiment.
Do not use final-ensemble threshold 0.47 for training checkpoint selection; historical
individual models use fixed threshold 0.5. The image list includes derived images;
parent grouping, not filename uniqueness, is the split-independence unit.
