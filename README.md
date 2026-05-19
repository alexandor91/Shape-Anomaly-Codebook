# Hierarchical Point–Patch Fusion with Adaptive Patch Codebook

PyTorch reference implementation of the paper *Hierarchical Point-Patch Fusion with
Adaptive Patch Codebook for 3D Shape Anomaly Detection* (CVPR 2026 supplementary).

The implementation faithfully follows Sections 3.1–3.6 of the main paper and the
codebook / RoPE-attention details in Sections 6.3–6.6 of the supplementary.

## Project layout

```
shape_anomaly_codebook/
├── README.md
├── requirements.txt
├── config.yaml             # All hyper-parameters
├── fps_grouping.py         # FPS + KNN/ball-query (pure PyTorch, no CUDA ext.)
├── augmentation.py         # 6 pseudo-anomaly types + GT offset generation
├── dataset.py              # Per-class PointCloudAnomalyDataset
├── encoder.py              # Lightweight PointNet++ stand-in for MinkUNet34C
├── patchify.py             # Adaptive multi-scale patchifier
├── codebook.py             # Multi-scale patch codebook (Alg. 1)
├── rope_attention.py       # 3-D Rotary PE + linear cross-attention (Eq. 4–5)
├── modulation.py           # Patch score modulation (Eq. 6–7)
├── network.py              # Full HierarchicalAnomalyNet
├── losses.py               # L_dist + λ_sim L_sim + λ_bce L_bce (Eq. 8–11)
├── train.py                # Training entry point
└── inference.py            # Inference + anomaly scoring
```

## Key design choices

- **Encoder backbone.** The paper uses a pre-trained Minkowski-UNet34C, which is
  hard to install. This repo ships a self-contained PointNet++-style encoder
  (`encoder.py::PointEncoder`) with the same output dim (32). If you have
  MinkowskiEngine installed, swap in `network.HierarchicalAnomalyNet(encoder=...)`.

- **Codebook as a `nn.Module` with buffers.** Features and counts live in
  `register_buffer`s so they save/load with `state_dict`. Spatial hashing uses a
  voxel grid (default 32³) over the canonical [−1, 1] cube.

- **RoPE for 3-D.** Standard RoPE is 1-D. We split the feature dim into three
  equal blocks (x, y, z) and apply rotary blocks per axis, then concatenate. This
  matches the spirit of Eq. (3) in the paper.

- **Linear attention.** Eq. (5) uses φ(x)=ψ(x)=elu(x)+1; we use the standard
  Katharopoulos formulation for O(N·D²) cost.

- **Scale selection.** We follow Eq. (2): pick the scale with maximal patch
  similarity sum (argmax over scales), not multi-scale fusion. The ablation
  in Table 6 reports 84.2% vs 81.2% in favour of max-selection.
## Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
conda install -c pytorch -c nvidia -c conda-forge pytorch=1.9.0 cudatoolkit=11.1 torchvision
conda install openblas-devel -c anaconda

# Uncomment the following line to specify the CUDA home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Or if you want a local MinkowskiEngine
cd lib
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

### Conda dependencies
```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch=1.9.0 cudatoolkit=11.1 torchvision
conda install openblas-devel -c anaconda

## Setup

```bash
pip install -r requirements.txt
```

Hardware: any CUDA GPU; the reference paper uses an RTX 3090 (≈2.1 GB at
N=10,000 points, batch 1).

## Data layout

Organise each class as a directory of point-cloud files. Each file is an Nx3 or
Nx6 (xyz, optional nxnynz) point cloud in `.npy` or `.ply` format.

```
data_root/
├── class_a/
│   ├── train/normal/0001.npy ...
│   └── test/
│       ├── good/0001.npy ...
│       └── anomaly/0001.npy + 0001_mask.npy ...
└── class_b/ ...
```

`*_mask.npy` is an N-dimensional binary array per anomaly sample (1 = anomalous
point) used for point-level AUC-ROC at evaluation time.

## Training

Train one class at a time (one codebook per class, as in Real3D-AD / Anomaly-ShapeNet):

```bash
python train.py --config config.yaml --class_name airplane --data_root /path/to/data
```

Training runs the two-phase procedure from Sec. 3 and Sec. 4.1:

1. **Phase 1** (≈first epoch): forward pass on normal samples populates the
   multi-scale patch codebook using Algorithm 1.
2. **Phase 2** (epochs 2..1500): negative augmentation generates pseudo-anomalous
   shapes; the full network is trained with the combined loss.

Checkpoints (model + codebook) are written to `runs/<class_name>/`.

## Inference

```bash
python inference.py \
    --checkpoint runs/airplane/best.pt \
    --input /path/to/test_shape.npy \
    --output preds.npy
```

Outputs a per-point anomaly score in `[0, 1]`. The script also retains the top-K
highest-scoring points as the final discrete detection set (Sec. 7.2; default
K=20, threshold 0.82).

## Reproducing paper numbers

The hyper-parameters in `config.yaml` match Sec. 4.1:

- Optimiser: Adam, lr 1e-3, 1500 epochs
- λ_sim = λ_bce = 0.5
- Patch scales (general): (32 patches × 64 pts), (64 × 32), (192 × 8)
- Patch scales (industrial): (8 × 192), (32 × 64), (64 × 32)
- Codebook merge threshold τ = 0.85
- Voxelisation grid 256³
- 10,000 points per sample

Switching to the industrial scales is a one-line config change.
