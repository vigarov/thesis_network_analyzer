# Network Analyzer — Visual-Context BTSP Pipeline

This repository implements the **visual context** experiments from a thesis on Behavioral Timescale Synaptic Plasticity (BTSP) in artificial neural networks. In neuroscience, BTSP describes rapid, durable learning events—often triggered by plateau potentials—that recruit neurons and teach them specific, lasting representations. Here, we ask whether similar recruitment-like phenomena can arise as a byproduct of ordinary backpropagation training, and whether the choice of optimizer shapes their frequency and character.

Concretely, the pipeline trains image-classification models under controlled **novelty paradigms** (label shuffles, sequential digit exposure, recover/reinforce schedules), records detailed optimizer and neuron telemetry throughout training, identifies BTSP-like **recruitment events**, scores them with a **BTSP Similarity Score (BSS)**, and generates the figures used in the report.

The work focuses on **MNIST** (a 5-layer fully connected network) and **CIFAR-10** (Inception or ResNet convolutional backbones with a shared FCN classification head), comparing five optimizers: SGD, AdaGrad, Adam, Pure Shampoo, and Grafted Shampoo. A typical workflow runs from optional pretraining, through simulation and analysis, to Jupyter notebooks that assemble the final plots.

---

## Setup

The project uses Python 3.12+ and [uv](https://docs.astral.sh/uv/) for dependency management. From the repository root:

```bash
# Install uv if you do not already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd network_analyzer
uv sync
```

You can also run `./setup.sh`, which performs `uv sync` and prints the installed PyTorch version.

PyTorch is resolved automatically via uv (CUDA wheels on Linux; CPU wheels are configured as a fallback). The Shampoo variants depend on a pinned revision of [facebookresearch/optimizers](https://github.com/facebookresearch/optimizers/tree/23b6ba547156ed5a3dd68a6f543e4858ed6ffbf6) on GitHub, declared in `pyproject.toml`. Most configs assume a GPU (`"device": "cuda"`); if you are working on CPU only, set `"device": "cpu"` in the relevant JSON config.

### Command-line tools

Three entry points cover the main stages of the pipeline:


| Command                  | Script                           | Purpose                                                      |
| ------------------------ | -------------------------------- | ------------------------------------------------------------ |
| `uv run compute`         | `scripts/run_simulation.py`      | Run training simulations and write raw results               |
| `uv run pretrain-models` | `scripts/pretrain_models.py`     | Train base or expert pretrain checkpoints                    |
| `uv run analyse`         | `scripts/analyse_sim_results.py` | Identify BTSP candidates, compute scores, and cache analysis |


---

## Project layout

The repository separates **experiment definitions**, **training infrastructure**, **analysis code**, and **local artifacts** (data, results, checkpoints) that are created when you run experiments:

```
network_analyzer/
├── input_configs/              # JSON experiment configs (MNIST + cifar10/)
├── experiments/mnist/          # Cat1 (global novelty) + Cat2 (sequential novelty)
├── models/                     # DNN5Hidden64, InceptionCIFAR, ResNetCIFAR
├── optimizers/                 # Per-optimizer signal extractors
├── compute_results/            # Training loop, checkpointing, pretrain
├── analysis/                   # I/O, BTSP scoring, FINAL analysis, report_plots/
│   └── notebooks/
│       └── REPORT_plots_FINAL.ipynb   # Thesis report figures
├── scripts/                    # CLI wrappers + train_many_seeds.py
├── data/                       # [created] MNIST / CIFAR auto-download (torchvision)
├── results/                    # [created] Raw simulation outputs
├── results_analysis/           # [created] Gzip-pickled analysis checkpoints
├── results_many_seeds/         # [created] Multi-seed period CSV exports
├── pretrained_models/          # [created] Base + expert pretrain checkpoints
└── local/save/plots/report/final/  # [created] Figures saved by report notebook
```

### Where datasets live

You do not need to download MNIST or CIFAR-10 manually. The first training or pretrain run fetches them via torchvision into `./data/`:


| Dataset  | Path under `./data/`   |
| -------- | ---------------------- |
| MNIST    | `MNIST/`               |
| CIFAR-10 | `cifar-10-batches-py/` |


The directories `data/`, `results*`, `pretrained_models/`, and `local/` are gitignored; they are expected to appear on your machine once you start running experiments.

### Where outputs are stored


| Artifact                     | Default path                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| Raw simulations              | `./results/<experiment_id>/<run_id>/<model_id>/`                                     |
| Analysis cache               | `./results_analysis/mnist/` or `./results_analysis/cifar/`                           |
| Multi-seed period CSVs       | `./results_many_seeds/<dataset>/`                                                    |
| Base pretrain checkpoints    | `pretrained_models/<dataset>/base/!OPT/!SD/`                                         |
| Expert models (for analysis) | `pretrained_models/<dataset>/expert/!OPT/` (CIFAR: `cifar10_!ARCH/expert/!OPT/!SD/`) |
| Report figures               | `local/save/plots/report/final/`                                                     |


Each simulation run is identified by `run_id`, the first six hex characters of a SHA-256 fingerprint of the training configuration. This makes it straightforward to tell runs apart even when they share the same experiment name.

---

## Experiments

The experimental design mirrors neuroscience novelty paradigms in a visual classification setting. **Cat1** experiments introduce novelty globally—typically by permuting labels across the dataset, so each training sample can be treated as its own trial. **Cat2** experiments introduce novelty sequentially: the model sees blocks of 100 samples from one digit at a time, which better matches trial-based protocols in the BTSP literature.

The table below maps the names used in the thesis to the corresponding `experiment_class` values and example MNIST config files under `input_configs/`:


| Thesis name                  | `experiment_class`             | Example config (MNIST)                       |
| ---------------------------- | ------------------------------ | -------------------------------------------- |
| Control global (no PT)       | `Cat1SampleShuffleControl`     | `cat1_sample_shuffle_control_multi.json`     |
| Control global (with PT)     | `Cat1GlobalPretrainControl`    | `cat1_global_pretrain_control_multi.json`    |
| Shuffle_all                  | `Cat1SampleShuffleFinetune`    | `cat1_sample_shuffle_finetune_multi.json`    |
| Shuffle_{0,1}                | `Cat1SampleShuffleConstrained` | `cat1_sample_shuffle_constrained_multi.json` |
| Shuffle_interleaved          | `Cat1SampleShuffleInterleaved` | `cat1_sample_shuffle_interleaved_multi.json` |
| Control sequential (no PT)   | `Cat2SequenceControl`          | `cat2_sequence_control_multi.json`           |
| Control sequential (with PT) | `Cat2SequencePretrainControl`  | `cat2_sequence_pretrain_control_multi.json`  |
| Shuffle_seq                  | `Cat2SequenceLabelPerm`        | `cat2_sequence_labelperm_multi.json`         |
| Recover                      | `Cat2SequenceRecover`          | `cat2_sequence_recover_multi.json`           |
| Reinforce                    | `Cat2SequenceReinforce`        | `cat2_sequence_reinforce_multi.json`         |


CIFAR-10 uses the same experiment classes with `InceptionCIFAR` or `ResNetCIFAR` as the model; configs are under `input_configs/cifar10/`.

**Models**

- `DNN5Hidden64` — five hidden layers of 64 units plus a 10-class head, used for MNIST.
- `InceptionCIFAR`, `ResNetCIFAR` — convolutional backbones feeding the same FCN head; the backbone is frozen after pretraining, so BTSP analysis focuses on the head layers.

**Typical settings** in the bundled configs include batch size 1, `checkpoint_cadence: "1 its"` (save neuron state every iteration), `num_trial_samples: 100` for sequential experiments, and evaluation on the last five samples per digit (50 images in total). Note that Adam's learning rate is derived from `base_lr` rather than set independently—for example, on MNIST, Adam uses approximately 0.0005 when `base_lr = 0.01`.

---

## Captured data and when

During training, the framework writes structured artifacts under each optimizer subdirectory. The analysis stage reads these files to locate BTSP candidates (points where a previously inactive neuron becomes assigned to a digit), characterize the resulting learning periods, and compute BSS components.

### Every training iteration

Because experiments use batch size 1, each training iteration corresponds to a single sample presentation.

**Optimizer signals** are collected at every step and flushed to `signals.npz` when the run finishes:


| Signal                  | Description                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| Gradients               | Full incoming weight gradient per tracked unit                                                        |
| `grad_norm`             | L2 norm of the gradient vector                                                                        |
| `grad_cosine_sim`       | Cosine similarity with the previous iteration's gradient                                              |
| Effective learning rate | Per-weight arrays for AdaGrad and Grafted Shampoo (`effective_lr__*`); constant for SGD (from config) |
| Adam moments            | First and second moment estimates (`exp_avg__*`, `exp_avg_sq__*`)                                     |
| Shampoo factors         | Kronecker inverse matrices (`h_inv__*__L/R`) and per-neuron `h_inv_norm` for Pure and Grafted Shampoo |


**Neuron state and metrics** are written to `neuron_timeseries.npz` and `training_metrics.npz` at each checkpoint (†):


| Signal         | Description                                                         |
| -------------- | ------------------------------------------------------------------- |
| Activations    | Pre-ReLU activations on 50 fixed eval digits (5 per class)          |
| Weights        | Full incoming weight vector per tracked unit                        |
| Train loss     | Running mean and standard error since the previous checkpoint       |
| Eval metrics   | Loss and accuracy per evaluation loader                             |
| Trial metadata | Trial names, trial-end checkpoint indices, global iteration indices |


† The config field `**checkpoint_cadence`** controls how often neuron state and metrics are saved. Production configs set this to `"1 its"` (every iteration). If you also want full weight snapshots of the entire model, set `**save_model_cp`: true**; these are written under `checkpoints/`.

### After training (derived CSVs)

Once a run completes, a short post-processing pass derives neuron-level labels from the saved activations:


| File                               | Description                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| `post_processing_neuron_digit.csv` | Per-neuron digit assignment (`assigned`, `partial`, `inactive`) at each checkpoint |
| `post_processing_dead.csv`         | Flags indicating whether a neuron is completely inactive across the eval set       |


### During analysis (computed from stored artifacts)

The analysis pipeline does not re-run training; it reconstructs higher-level quantities from the files above:


| Metric                      | Description                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| BTSP periods                | Intervals from neuron recovery (dead → assigned) through loss of assignment                                  |
| Activation angle            | Spontaneity score from the pre-assignment activation ramp                                                    |
| Period length               | Duration of stable digit assignment                                                                          |
| Robustness                  | Representational consistency (CW-SSIM for FCN sensitivity maps; Grad-NAM + RDX vs an expert model for CIFAR) |
| Sensitivity / Grad-NAM maps | Pixel-wise or convolutional receptive-field heatmaps at period start                                         |
| ΔW vicinity                 | Weight-change patterns around `t_start`                                                                      |
| Effective LR trajectories   | Network-wide summaries derived from stored optimizer signals                                                 |


### Result tree (per run)

For reference, a single completed run typically looks like this:

```
results/<experiment_id>/<run_id>/<model_id>/
├── config.json
├── metadata.json
└── optimizers/<optimizer_id>/
    ├── signals.npz
    ├── neuron_timeseries.npz
    ├── training_metrics.npz
    ├── post_processing_neuron_digit.csv
    ├── post_processing_dead.csv
    └── optimizer_config.json
```

---

## Reproduction steps

The following sections walk through reproducing results from scratch. Depending on which experiment you choose, you may not need every step—for instance, experiments without pretraining skip step 2 (base pretrain), though expert models (step 3) are still required for robustness scoring during analysis.

Simulation outputs dominate disk usage: a single MNIST experiment with all five optimizers is on the order of **10–15 GB**, and reproducing the full visual-context study (all configs, optimizers, and seeds) requires roughly **~400 GB** in total. The estimates below are indicative; exact sizes depend on iteration count, optimizer (Shampoo variants store large preconditioner histories), and whether you keep both raw results and analysis caches.

### 1. Datasets

There is no separate download command. The first invocation of `compute` or `pretrain-models` will populate `./data/` with MNIST and/or CIFAR-10 as needed.

**Disk space:** ~65 MB for MNIST alone; ~180 MB for CIFAR-10 (extracted batches). If both datasets are used over the course of the project, expect **~400 MB** in `./data/` (torchvision may also leave a one-time download tarball for CIFAR-10 until removed).

### 2. Base pretrain

Many experiments begin from a pretrained model rather than random initialization. If your config sets `"initial_model_mode": "pretrain"`, you will need matching checkpoints under `pretrained_models/.../base/` before running the main simulation. The pretrain script trains until a test-accuracy threshold is reached:

```bash
uv run pretrain-models \
  --config input_configs/cat2_sequence_pretrain_control_multi.json \
  --output-dir pretrained_models/mnist/base/!OPT/!SD/
```

The output path uses template tokens: `!OPT` is replaced by the optimizer slug (e.g. `sgd_lr0.01`), and `!SD` by the seed directory.

**Disk space:** Each checkpoint bundle is small compared to simulation outputs: `model.pt`, `optimizer.pt`, and JSON metadata. Roughly **~1–5 MB** per (optimizer, seed) for MNIST; **~15–40 MB** for CIFAR Inception/ResNet. A full MNIST base-pretrain matrix (5 optimizers, one seed) is **~25 MB**; scaling to seven seeds is **~175 MB**. Both CIFAR architectures together add **~200–400 MB** per seed.

### 3. Expert models

Robustness scoring compares a neuron's sensitivity maps during a BTSP period against those of a separately trained **expert** model. These checkpoints must exist before running analysis:

```bash
uv run pretrain-models \
  --config input_configs/cat2_sequence_pretrain_control_multi.json \
  --expert \
  --output-dir pretrained_models/mnist/expert/!OPT/
```

By default, the analysis scripts look for experts at the paths defined in `analysis/final/final_analysis_config.py`:

- MNIST: `pretrained_models/mnist/expert/!OPT/`
- CIFAR-10: `pretrained_models/cifar10_!ARCH/expert/!OPT/!SD/`

**Disk space:** Same order of magnitude as base pretrain checkpoints—the expert preset trains on more data (20k MNIST / 40k CIFAR samples) but saves the same model files, not full training traces. Expect **~25 MB** for all five MNIST experts (one seed) and **~200–400 MB** across both CIFAR architectures.

### 4. Run simulations

The main training step is `compute`. You can restrict to a single optimizer for a quick test, run all optimizers listed in the config, or sweep over seeds:

```bash
# Single optimizer — useful for smoke testing
uv run compute \
  --config input_configs/cat2_sequence_recover_multi.json \
  --optimizer sgd \
  --output-dir ./results

# All optimizers (when the config has "optimizer": "all")
uv run compute \
  --config input_configs/cat2_sequence_recover_multi.json \
  --output-dir ./results

# Multiple seeds
uv run compute \
  --config input_configs/cat2_sequence_recover_multi.json \
  --multi_seed 6,7,14
```

If results for a given configuration already exist, add `--force` to overwrite them. Note that if an existing `config.json` in the output directory does not match your requested configuration, the run is rejected rather than silently mixed—either adjust the config or point `--output-dir` elsewhere.

For CIFAR-10, the command is the same in structure; only the config path changes:

```bash
uv run compute \
  --config input_configs/cifar10/cat2_sequence_recover_resnet_multi.json \
  --output-dir ./results
```

**Disk space:** This step produces the bulk of project storage under `./results/`. Each run writes per-optimizer `signals.npz` and `neuron_timeseries.npz` files; with `checkpoint_cadence: "1 its"`, both grow linearly with training length. For a typical sequential MNIST experiment (~4k iterations, e.g. Recover):

| Optimizer | Approx. size per run |
|-----------|---------------------|
| SGD / AdaGrad | ~1 GB |
| Adam | ~2 GB |
| Pure / Grafted Shampoo | ~5 GB each (Kronecker factor history) |

All five optimizers on one MNIST experiment: **~10–15 GB**. A single-optimizer smoke test: **~1–5 GB**. Reproducing the complete visual-context dataset used in the thesis (all MNIST and CIFAR configs, five optimizers, multiple seeds) requires roughly **~400 GB** cumulative in `./results/` if everything is retained.

### 5. Analyse

Analysis scans the raw results tree, identifies recruitment events, computes BSS and related metrics, and writes compressed checkpoints for the notebooks:

```bash
uv run analyse \
  --dataset mnist \
  --input-dir ./results \
  --output-dir ./results_analysis/mnist
```

For CIFAR-10, use `--dataset cifar10` and `--output-dir ./results_analysis/cifar`. Output files land under `results_analysis/<dataset>/<sha10>/`.

**Disk space:** Analysis does not duplicate the raw NPZ archives; it adds gzip-pickled summaries (`results.pkl.gz`) with period tables and derived scores. These are comparatively small—typically **~1–30 MB per analysed (experiment, optimizer, seed) run**. A full analysis cache over a complete results tree is usually **~1–10 GB**, depending on how many simulations you have under `./results/`.

### 6. Multi-seed exports (optional)

Several report figures (especially MNIST error bars) aggregate statistics across multiple random seeds. The helper script below trains a set of configs and exports compact period CSVs without keeping full simulation trees:

```bash
uv run python scripts/train_many_seeds.py \
  --config-dir input_configs \
  --seeds 6,7,14,21,28,35,42 \
  --output-root ./results_many_seeds
```

You can limit which experiments are included with `--exp cat2_sequence_recover,cat2_sequence_reinforce`.

**Disk space:** The script keeps only compact CSV exports under `./results_many_seeds/` (period tables and run summaries), not full simulation trees. A broad MNIST sweep (all configs, seven seeds, five optimizers) is typically **~50–500 MB** of persistent CSV data. During each job, transient training artifacts are written to `<output-root>/scratch/` and deleted afterward—you still need enough free space for the largest single simulation (~**5–15 GB** peak) while that job runs.

### 7. Report figures

The notebook `[analysis/notebooks/REPORT_plots_FINAL.ipynb](analysis/notebooks/REPORT_plots_FINAL.ipynb)` assembles the curated figures for the thesis report. It reads from `results_analysis/` and, where applicable, `results_many_seeds/`, and saves PDF/PNG output to `local/save/plots/report/final/`. Figures include BTSP period counts, optimizer profiles, activation angles, layer-wise BTSP/BSS trends, novelty-response comparisons, training losses, and convergence thresholds.

For more exploratory views of individual experiments, two companion notebooks go deeper:

- `analysis/notebooks/FINAL_compare_all_optimizers_mnist.ipynb` / `FINAL_compare_all_optimizers_cifar.ipynb`
- `analysis/notebooks/FINAL_additional_plots_mnist.ipynb` / `FINAL_additional_plots_cifar.ipynb`

**Disk space:** The report notebook writes PDF/PNG figures to `local/save/plots/report/final/`. Expect **~50–200 MB** for the curated thesis figure set. Running the companion notebooks and saving all outputs may add another **~100 MB–1 GB**.

---

## Worked example: MNIST Recover with SGD

The **Recover** experiment is a good starting point: it pretrains the model, exposes it to a digit with correct labels, then mislabels the same images, and finally shows the correct digit again—mirroring a perturb-and-recover novelty schedule. The commands below run the full pipeline for SGD from a fresh clone:

```bash
uv sync

# Base pretrain (Recover expects initial_model_mode: pretrain)
uv run pretrain-models \
  --config input_configs/cat2_sequence_recover_multi.json \
  --optimizer sgd \
  --output-dir pretrained_models/mnist/base/!OPT/!SD/

# Expert model for robustness scoring
uv run pretrain-models \
  --config input_configs/cat2_sequence_recover_multi.json \
  --optimizer sgd \
  --expert \
  --output-dir pretrained_models/mnist/expert/!OPT/

# Main experiment
uv run compute \
  --config input_configs/cat2_sequence_recover_multi.json \
  --optimizer sgd \
  --output-dir ./results

# BTSP identification and scoring
uv run analyse \
  --dataset mnist \
  --input-dir ./results \
  --output-dir ./results_analysis/mnist

# Open analysis/notebooks/REPORT_plots_FINAL.ipynb and run all cells
```

For this single-optimizer example, expect roughly **~1 GB** in `./results/` (one MNIST Recover run), **~5 MB** of pretrain checkpoints, and **~10–50 MB** in `./results_analysis/mnist/` after analysis.

---

## Configuration reference

Each experiment is described by a JSON file in `input_configs/`. The snippet below illustrates the common structure (from `cat2_sequence_pretrain_control_multi.json`):

```json
{
  "experiment_class": "Cat2SequencePretrainControl",
  "experiment_config": {
    "pretrain_on_k_samples": 1000,
    "num_trial_samples": 100
  },
  "model_class": "DNN5Hidden64",
  "model_config": { "activation": "relu" },
  "optimizer": "all",
  "trial_epochs": 1,
  "base_lr": 0.01,
  "batch_size": 1,
  "seed": 3003,
  "checkpoint_cadence": "1 its",
  "device": "cuda",
  "experiment_runs": 10,
  "initial_model_mode": "pretrain",
  "use_initial_model": "pretrained_models/mnist/base/"
}
```


| Field                | Role                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `experiment_class`   | Registered experiment name (see table above)                                             |
| `experiment_config`  | Experiment-specific parameters such as `num_trial_samples` or digit constraints          |
| `model_class`        | `DNN5Hidden64`, `InceptionCIFAR`, or `ResNetCIFAR`                                       |
| `optimizer`          | A single shorthand, a comma-separated list, or `"all"` to run every registered optimizer |
| `base_lr`            | Base learning rate; Adam's effective rate is scaled from this value                      |
| `seed`               | Random seed, included in the run fingerprint                                             |
| `checkpoint_cadence` | How often neuron state is saved (`"1 its"` = every iteration)                            |
| `initial_model_mode` | `"init"` for random weights, or `"pretrain"` to load from `use_initial_model`            |
| `use_initial_model`  | Path template pointing to base pretrain checkpoints                                      |
| `experiment_runs`    | Number of repetitions (for sequential experiments, this cycles the starting digit)       |


CIFAR configs additionally set `"dataset": "cifar10"`; see `input_configs/cifar10/` for examples.