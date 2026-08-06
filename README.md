# fairopt-causal — Causal Optimal Transport Experiments

Optimal Transport for **causal** recourse analysis: three experiments (C1-C3) that
study how the true latent/causal transport differs from a naive ambient transport.

## Experiments

| ID | Module | What it measures | Output CSV |
|----|--------|------------------|------------|
| C1 | `c1_causality` | Stability of OT under anisotropic deformations `X = M U`, `M = I + cN`. Tracks conjugacy error `‖T_X − M∘T_U∘M⁻¹‖` and its power-law scaling in the anisotropy ratio `κ = Λ/λ` (2D and 5D) | `results/c1_causality_2d.csv`, `results/c1_causality_5d.csv` |
| C2 | `c2_recourse_causal` | The cost of causal ignorance: Standard OT vs Causal OT over adult, LSAC, COMPAS, German. True Intervention Effort (latent L2) vs Apparent Cost (ambient L2), mean ± std over 5-fold CV | `results/c2_recourse_causal.csv` |
| C3 | `c3_causality_real` | Causal OT on the **ecoli70** Gaussian Bayesian network (`run_ecoli70`) and PCA-conjugacy deviation on lsac / student / credit_default (`run`) | `results/c3_causality_ecoli70.csv`, `results/c3_causality_real.csv` |

## Project structure

```
causal/
├── fairopt/                  # the package (self-contained)
│   ├── core/                 # Sinkhorn solver, cost, metrics
│   ├── data/                 # dataset loaders + preprocessing
│   ├── utils/                # ILR transform
│   └── experiments/          # c1_causality, c2_recourse_causal,
│                             # c3_causality_real, config, runner, run_all
├── tests/                    # pytest suite (sinkhorn, ilr)
├── visualization.ipynb       # C1, C2, C3 plots + LaTeX tables
├── pyproject.toml            # dependencies
├── Dockerfile                # reproducible container
└── Makefile                  # convenience targets
```

## Installation

Requires Python >= 3.10.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[viz]"          # -e . for experiments only
```

The `viz` extra adds Jupyter, matplotlib and seaborn for `visualization.ipynb`.

## Running the experiments

Each experiment writes CSVs into `results/` (created automatically).

```bash
# single experiments
python -m fairopt.experiments.c1_causality
python -m fairopt.experiments.c2_recourse_causal
python -m fairopt.experiments.c3_causality_real            # run() — real datasets
python -m fairopt.experiments.c3_causality_real           # __main__ — run_ecoli70()
python -c "from fairopt.experiments.c3_causality_real import run, run_ecoli70; run(); run_ecoli70()"

# everything sequentially
python -m fairopt.experiments.run_all
```

## Visualization

```bash
jupyter notebook visualization.ipynb
```

The notebook loads the CSVs from `results/` and renders the figures/tables for
C1, C2 and C3. Run the experiments first (or the `--help` of `run_all`).

## Docker (reproducible environment)

```bash
make docker-build          # docker build -t fairopt-causal .
make docker-run            # runs all experiments, results written to ./results
```

or manually:

```bash
docker build -t fairopt-causal .
docker run --rm -v "$(pwd)/results:/app/results" fairopt-causal
```

Override the default command to run a single experiment or the notebook server:

```bash
docker run --rm -v "$(pwd)/results:/app/results" fairopt-causal python -m fairopt.experiments.c1_causality
docker run --rm -p 8888:8888 -v "$(pwd)/results:/app/results" fairopt-causal jupyter notebook --ip=0.0.0.0 --allow-root
```

## Reproducibility notes

- **Network access is required on first run**: OpenML datasets (adult, LSAC,
  credit_default, student, german), the COMPAS CSV from GitHub, and the ecoli70
  network (`https://www.bnlearn.com/bnrepository/ecoli70/ecoli70.rda`, parsed with
  `rdata`). Nothing is vendored; the same URLs are pinned in the loaders.
- Randomness is seeded (`torch` seed 42, `np.random.RandomState(42)`) inside each
  experiment; C2/C3 additionally use sklearn's `KFold(random_state=42)`.
- Results are deterministic for a fixed seed set, but exact numeric output can
  differ slightly across platforms due to BLAS/threading.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v --tb=short
```
