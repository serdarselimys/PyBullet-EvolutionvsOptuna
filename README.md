![Screenshot](2.jpg)

# PyBullet Gait Optimization: Genetic Algorithm vs Optuna

Two parallel gait-optimization pipelines for a 6-legged robot (**HexaDog**) simulated in PyBullet, plus a live TensorBoard dashboard for watching (and comparing) both approaches while they train.

- **`Evolution_Optimization.py`** — optimizes leg-gait parameters (step amplitude, frequency) with a custom Genetic Algorithm.
- **`Optuna_Optimization.py`** — optimizes the same parameters with [Optuna](https://optuna.org/) (TPE or CMA-ES samplers).
- Both scripts run many experiments in parallel (`ProcessPoolExecutor`) across gait modes (`straight`, `sideway`, `diagonal`, `spin`), and both stream live per-generation results to a **single shared TensorBoard dashboard**, so `GA_*` and `Optuna_*` runs sit side by side for direct comparison.

## Repository contents

```
.
├── Evolution_Optimization.py   # Genetic Algorithm optimizer
├── Optuna_Optimization.py      # Optuna (TPE / CMA-ES) optimizer
├── dashboard_logger.py         # Per-run TensorBoard/tensorboardX scalar+histogram logger
├── live_queue_dashboard.py     # Combines repeats of the same config live, via a multiprocessing Queue
├── aggregate_dashboard.py      # Optional: post-hoc / alternative aggregation of separate per-repeat runs
├── launch_dashboard.py         # Convenience launcher for TensorBoard
├── HexaDog_ZBD.urdf            # Robot model
└── meshes/                     # STL meshes referenced by the URDF
```

## 1. Clone the repository

```bash
git clone https://github.com/serdarselimys/PyBullet-EvolutionvsOptuna.git
cd PyBullet-GeneticAlgorithm-EvolutionvsOptuna
```

## 2. Install dependencies

Python 3.9+ is recommended. Using a virtual environment is optional but encouraged:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Notes:
- `optuna` is only required by `Optuna_Optimization.py`.
- The dashboard works with either `tensorboardX` (scalars **and** histograms) or plain `tensorboard` alone (scalars only, used automatically as a fallback if `tensorboardX` isn't installed). Installing both is the safest option.
- If you skip the TensorBoard packages entirely, both optimizers still run fine — dashboard logging just silently becomes a no-op.

## 3. Run an optimization

Both scripts are configured by editing the constants at the top of the file (no CLI flags) — open the script and adjust:

- `MODES` — which gait modes to optimize: `"straight"`, `"sideway"`, `"diagonal"`, `"spin"` (any subset).
- `NUM_ROBOTS`, `GAIT_TRIALS` — population size / number of generations (Optuna: trials per generation / number of generations).
- `NUM_EXPERIMENTS` — how many repeats of each config to run (repeats are combined live on the dashboard).
- `RENDER_MODE` — `"headless"` (background, needed for parallel runs) or `"windowed"` (opens the PyBullet GUI, single run only).
- `TARGET_SPEEDS` / `TARGET_ROT_SPEEDS`, `BODY_HEIGHTS`, `STEP_HEIGHTS`, `DIRECTIONS` / `DIAGONAL_DIRECTIONS` — the parameter sweep. Leave a direction list empty (`[]`) to skip its associated modes.
- `ENABLE_DASHBOARD` / `DASHBOARD_LOGDIR` — toggle and destination (`"runs"` by default) for live TensorBoard logging.
- `Optuna_Optimization.py` only: `SAMPLER_MODE` — `"tpe"` (default) or `"cmaes"`.

Run whichever optimizer you want from the repo root:

```bash
python Evolution_Optimization.py
```

```bash
python Optuna_Optimization.py
```

Each spawns `NUM_EXPERIMENTS` × (modes × directions × ... ) parallel worker processes (one per CPU core minus one, via `ProcessPoolExecutor`), simulates every generation in PyBullet, and prints per-generation progress to the terminal via `tqdm`.

On completion:
- `Evolution_Optimization.py` writes results to **`ga_gait_results.csv`**.
- `Optuna_Optimization.py` writes results to **`gait_results.csv`**.

Each CSV contains one row per experiment with the best score, winning DNA (`step_amplitude`, `frequency`), and best-run quality metrics.

You can run both scripts back-to-back (or even leave `ENABLE_DASHBOARD = True` in both) to build up a shared `runs/` directory and compare GA vs Optuna on the same dashboard.

## 4. Launch the live TensorBoard dashboard

While an optimization is running (or after it finishes), launch TensorBoard against the `runs/` directory:

```bash
python launch_dashboard.py
```

This starts TensorBoard on `http://localhost:6006` and opens it in your browser automatically. Optional flags:

```bash
python launch_dashboard.py --logdir runs --port 6006 --no-browser
```

Or run TensorBoard directly:

```bash
tensorboard --logdir runs
```

### What you'll see

- One run per config group, named e.g. `GA_straight_dir1_h0.2` or `Optuna_TPE_straight_dir1_h0.2` — **not** one run per individual repeat. `live_queue_dashboard.py` combines all `NUM_EXPERIMENTS` repeats of a config live, in the main process, as workers report in, so nothing extra is written to disk.
- Curves for `score/best_this_gen`, `score/mean_this_gen`, `score/worst_this_gen`, `score/all_time_best`, plus `dna/*` (winning gene values) and `metrics/*` (best-run quality metrics), updating generation by generation while the optimizer is still running.
- Because GA runs are prefixed `GA_` and Optuna runs are prefixed `Optuna_`, both show up together in the same TensorBoard for direct side-by-side comparison.

### Optional: `aggregate_dashboard.py`

`live_queue_dashboard.py` (used automatically by both optimizer scripts) is the recommended path — it combines repeats live with nothing extra written to disk. `aggregate_dashboard.py` is provided as an **alternative/legacy** approach for cases where each repeat is logged to its own run directory on disk (e.g. `..._exp0`, `..._exp1`, ...) and you want to combine them into a `<config>_MEAN` run afterwards:

```bash
python aggregate_dashboard.py --logdir runs --min-repeats 2
```

It can also be run as a background thread (`LiveDashboardAggregator`) if you're wiring up your own experiment scripts that log per-experiment runs directly via `dashboard_logger.DashboardLogger` instead of the queue-based approach.

## How the two optimizers compare

Both scripts share the same PyBullet simulation, inverse kinematics, and scoring logic — only the search strategy for the two genes (`step_amplitude`, `frequency`) differs:

| | Evolution_Optimization.py | Optuna_Optimization.py |
|---|---|---|
| Search strategy | Custom Genetic Algorithm (tournament selection, crossover, mutation, elitism) | Optuna (TPE or CMA-ES sampler) |
| "Generation" | GA generation (population evolves) | Optuna study `ask`/`tell` round (`NUM_ROBOTS` trials per round) |
| Output CSV | `ga_gait_results.csv` | `gait_results.csv` |
| Dashboard run prefix | `GA_...` | `Optuna_TPE_...` / `Optuna_CMAES_...` |

This makes it straightforward to run identical parameter sweeps through both scripts and compare convergence speed and final gait quality side-by-side in TensorBoard.
