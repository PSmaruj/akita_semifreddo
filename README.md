# Akita Semifreddo: Designing DNA Sequences for Desired 3D Genome Folding

A sequence design pipeline combining [Ledidi](https://github.com/jmschrei/ledidi) with a half-frozen Akita wrapper (Semifreddo) to efficiently optimize DNA sequences that produce desired three-dimensional genome folding patterns.

## Overview

This repo introduces **Semifreddo**, a "half-frozen" wrapper around the [PyTorch Akita V2](https://github.com/PSmaruj/akita_pytorch) Hi-C prediction model. Instead of passing the full 1.3 Mb sequence through the convolutional tower on every optimization step, Semifreddo caches trunk activations and recomputes only the small window around the edited bin(s), giving >3× lower peak memory and ~25% faster runtime while producing identical predictions (Pearson R = 1.0).

Sequence optimization is performed by [Ledidi](https://github.com/jmschrei/ledidi), which proposes discrete nucleotide edits guided by a masked local L1 loss over the predicted contact map. Designs are validated independently using [AlphaGenome](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/).

Five design experiments are performed across 8 cross-validation folds in mouse mESC Hi-C data (Hsieh 2019, mm10):
- **Boundaries** — create new TAD boundaries in flat genomic regions, with CTCF insertion allowed
- **Boundaries (no CTCF)** — create new boundaries while penalizing CTCF motif insertion via a FIMO-based loss term, testing whether boundaries can be designed without recruiting CTCF
- **Boundary suppression** — weaken existing strong boundaries by optimizing away from the current insulation state, with CTCF motifs in the original sequence frozen (masked) during optimization
- **Dots** — create chromatin loops (focal contact enrichment) by simultaneously editing two anchor bins flanking the dot position
- **Flames** — create stripe-like contact patterns by editing a single central bin

## Installation

### Option A — pip

If you want to use the `semifreddo` and `utils` modules in your own code
without modifying them:

```bash
pip install git+https://github.com/PSmaruj/akita_semifreddo.git
```

> **Note:** The pip installation does not include the tutorial scripts,
> analysis notebooks, or optimization code. If you want to run the tutorials
> or modify the source code, please clone the repository (Option B).

### Option B — clone (recommended for tutorials and development)

```bash
git clone https://github.com/PSmaruj/akita_semifreddo.git
cd akita_semifreddo

conda env create -f environment.yml
conda activate akita_semifreddo
```

> **Note:** The default environment is configured for GPU systems (CUDA 11.8).
> If you are on a CPU-only system, see the comments in
> [`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt)
> for instructions on switching to the CPU-only PyTorch build.

### Installing PyTorch Akita

Both options require the PyTorch Akita V2 model. Clone it separately:

```bash
git clone https://github.com/PSmaruj/akita_pytorch.git
```

The model class can also be pip-installed on its own:

```bash
pip install git+https://github.com/PSmaruj/akita_pytorch.git
```

### Downloading Model Weights

The fine-tuned PyTorch AkitaV2 model weights are hosted on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19599537.svg)](https://doi.org/10.5281/zenodo.19599537)

**The weights are required to reproduce the analyses from the paper and to run the tutorials.** To download and extract them, run from the root of the `akita_pytorch` directory:

```bash
wget -O akita_pytorch_model_weights.tar.gz https://zenodo.org/record/19599537/files/akita_pytorch_model_weights.tar.gz
tar -xzf akita_pytorch_model_weights.tar.gz && rm akita_pytorch_model_weights.tar.gz
```

This will recreate the `models/` directory with the following structure:

```
models/
├── finetuned/
│   ├── mouse/<dataset>/checkpoints/*.pth
│   └── human/<dataset>/checkpoints/*.pth
└── trained_from_scratch/
    └── mouse/<dataset>/checkpoints/*.pth
```

### Using AkitaSF together with AkitaPT (recommended)

If you plan to use AkitaSF with AkitaPT as described in the paper, we recommend
setting up a single shared environment based on AkitaPT's `environment.yml` and
installing the additional AkitaSF dependencies on top:

```bash
# Clone both repositories
git clone https://github.com/PSmaruj/akita_pytorch.git
git clone https://github.com/PSmaruj/akita_semifreddo.git

# Create and activate the AkitaPT environment
cd akita_pytorch
conda env create -f environment.yml
conda activate pytorch_akita

# Install the additional AkitaSF dependencies
pip install memelite==0.2.0 tangermeme==1.0.0 ledidi==2.1.0 seqpro==0.9.0 tqdm==4.67.1
```

### AlphaGenome (optional — validation only)

The cross-model validation notebooks (`analysis/alpha_genome_validation/`)
require AlphaGenome, which is not available on PyPI and must be set up
separately:

1. Follow the installation instructions at the [AlphaGenome repository](https://github.com/google-deepmind/alphagenome)
2. AlphaGenome requires its own conda environment — we recommend keeping
   it separate from the main `akita_semifreddo` environment and running the
   validation notebooks within it
3. Obtain your own API key from Google and set it in the notebooks:
   ```python
   API_KEY = "YOUR_ALPHAGENOME_API_KEY"  # replace with your own key
   ```
   Never commit your API key to a public repository.

> **Note:** AlphaGenome validation is not required to run the core AkitaSF
> sequence design pipeline. All optimization scripts and tutorials work
> with the main `akita_semifreddo` environment only.


## Repository Structure

```
akita_semifreddo/
│
├── 📁 semifreddo/                  # Half-frozen Akita wrapper and optimization building blocks
│   ├── __init__.py
│   ├── semifreddo.py               # Semifreddo, SemifreddoLedidiWrapper, TwoAnchorSemifreddoLedidiWrapper,
│   │                               # CTCFAwareSemifreddoWrapper, MultiBinSemifreddoLedidiWrapper,
│   │                               # StackingDesignWrapper
│   ├── losses.py                   # LocalL1Loss, LocalL1LossWithCTCFPenalty, MultiHeadLocalL1Loss
│   └── optimization_loop.py        # run_one_design, run_one_design_dot, run_fold, and utilities
│
├── 📁 utils/                       # Shared utilities
│   ├── __init__.py
│   ├── data_utils.py               # OHE encoding, matrix utilities, GC content
│   ├── dataset_utils.py            # PyTorch Dataset classes for sequences and contact maps
│   ├── df_utils.py                 # DataFrame utilities for loading and summarizing results
│   ├── fimo_utils.py               # FIMO-based motif scanning and PWM scoring
│   ├── insulation_utils.py         # Insulation score computation and flat region detection
│   ├── model_utils.py              # Model loading, inference, and target generation
│   ├── plot_utils.py               # Contact map and CTCF track visualization
│   └── scores_utils.py             # Boundary, dot, and flame scoring functions
│
├── 📁 optimizations/               # Optimization scripts and results
│   ├── boundaries/                 # Boundary creation designs (CTCF insertion allowed)
│   ├── boundaries_no_ctcf/         # Boundary creation designs (no CTCF insertion)
│   ├── boundary_suppression/       # Boundary weakening designs
│   ├── cross_celltype_boundaries/  # Human sequences targeting H1hESC and HFF simultaneously (unvalidated)
│   ├── dots/                       # Chromatin loop (dot) designs
│   ├── flames/                     # Stripe (flame) designs
│   ├── fountains/                  # Fountain designs (unvalidated)
│   └── genome_rewiring/            # Large-scale genome rewiring experiments
│
├── 📁 analysis/                    # Analysis notebooks and scripts
│   ├── akita_vs_semifreddo/        # Semifreddo correctness and speedup validation
│   ├── alpha_genome_validation/    # Independent validation using AlphaGenome
│   ├── backgrounds/                # Background (shuffled) sequence generation
│   ├── flat_regions/               # Flat genomic region identification pipeline
│   └── natural_features/           # Natural boundary, dot, and flame baselines
│
├── 📁 data/                        # Reference data
│   ├── ctcf_tables/                # CTCF binding site tables
│   ├── data_driven_dot/            # Data-driven dot patch used to construct the dot mask
│   ├── pwm/                        # Position weight matrices (JASPAR CTCF MA0139.1)
│   └── sine_b2_tables/             # SINE B2 element tables
│
├── 📁 tutorial/                    # Step-by-step worked examples
│   ├── tutorial_01_boundary_design.ipynb   # End-to-end boundary design
│   ├── tutorial_02_dot_design.ipynb        # Chromatin loop (dot) design
│   ├── tutorial_03_flame_design.ipynb      # Stripe (flame) design
│   └── tutorial_04_custom_target.ipynb       # Custom contact map target design (smiley face)
│
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Tutorials

Four tutorials are provided in `tutorial/`, written as annotated Python scripts
with cells delimited by `# %%` comments (compatible with Jupyter, VS Code, and
Spyder). Each tutorial is self-contained: it generates all required intermediate
files from scratch, saves them to a local `tmp_data/` directory, and removes
them at the end. Before running any tutorial, make sure your environment matches
`environment.yml` / `requirements.txt` and update the path constants at the top
of each script.

- **`tutorial_01_boundary_design.ipynb`** — end-to-end walkthrough of designing a
  strong TAD boundary in a flat genomic region. Covers one-hot encoding,
  boundary mask construction, tower output caching, `SemifreddoLedidiWrapper`
  setup and sanity-checking, Ledidi optimisation, CTCF motif analysis, and
  result visualisation. Recommended starting point.

- **`tutorial_02_dot_design.ipynb`** — designing a chromatin loop (dot) using a
  data-driven 15×15 Hi-C pileup patch (provided in `data/`) as the target.
  Introduces `TwoAnchorSemifreddoLedidiWrapper`, which optimises two anchor
  bins simultaneously — one per loop anchor.

- **`tutorial_03_flame_design.ipynb`** — designing a chromatin stripe (flame) by
  stamping an analytical 'L'-shaped stripe mask onto the contact map. Uses the
  same single-anchor `SemifreddoLedidiWrapper` as Tutorial 1.

- **`tutorial_04_custom_target.ipynb`** — designing a sequence toward a fully
  arbitrary custom contact map target (a smiley face). Demonstrates that the
  only requirement for a custom AkitaSF design is a mask function that converts
  an idea into an upper-triangular index vector. Results should be interpreted
  critically and assessed visually.

## Key Design Choices

**Semifreddo efficiency.** The convolutional tower of Akita V2 has a receptive field of ~10 kb (~5 bins at 2048 bp/bin). Semifreddo exploits this by recomputing only an 11-bin window around the edited bin on each optimization step, reducing tower computation from 640 bins to 11 bins per step.

**Local masked loss.** `LocalL1Loss` computes L1 loss only over the contact map positions where the target feature is expected, scaled by the inverse of the mask's coverage fraction to keep loss magnitudes comparable across mask sizes. The mask can be extended beyond the feature itself to implement a semi-local loss that simultaneously constrains neighboring regions.

**CTCF penalty.** `LocalL1LossWithCTCFPenalty` adds a FIMO-based penalty term (γ × Σ FIMO scores) to discourage CTCF motif insertion in the edited bin, used for boundary designs that should not rely on CTCF recruitment.

**Cross-validation.** All designs are run across 8 folds of the Akita V2 train/validation split, with held-out models used as independent validators within Akita, and AlphaGenome used as a fully independent external validator.

## Citation

If you use Akita Semifreddo in your research, please cite:

**Akita Semifreddo**:
```
[PLACEHOLDER — preprint citation]
```

**Akita V2**:
```
Smaruj PN, Kamulegeya F, Kelley DR, Fudenberg G (2025)
Interpreting the CTCF-mediated sequence grammar of genome folding with Akita v2.
PLOS Computational Biology 21(2): e1012824.
https://doi.org/10.1371/journal.pcbi.1012824
```

**Ledidi**:
```
Schreiber J, Lorbeer FK, Heinzl M, Reiter F, Rafanel B, Lu YY, Stark A, Noble WS (2025)
Programmatic design and editing of cis-regulatory elements. bioRxiv.
https://doi.org/10.1101/2025.04.22.650035
```

## Contact

Feedback and questions are welcome. Please contact:

- **Geoffrey Fudenberg**: fudenber at usc dot edu
- **Paulina Smaruj**: smaruj at usc dot edu

Or open an issue on GitHub.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Related Resources

- [PyTorch Akita v2](https://github.com/PSmaruj/akita_pytorch)
- [Ledidi](https://github.com/jmschrei/ledidi)
- [Original Akita (Nature Methods)](https://www.nature.com/articles/s41592-020-0958-x)
- [AkitaV2 (PLOS Computational Biology)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012824)
- [AlphaGenome](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/)

---

*Last updated: April 2026*