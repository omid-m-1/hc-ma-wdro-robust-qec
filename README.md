# Package 10 — run instructions

## 1. Unzip

Run each command separately in Bash:

```bash
unzip qec_robust_study_v10_source.zip
```
```bash
cd qec_robust_study_v10_source
```
```bash
mkdir -p logs
```

## 2. Select a Python environment

Activate your working server environment:

```bash
conda activate astra311
```

For a new installation instead, create a separate Python3.11 environment and run:

```bash
python3.11 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
python -m pip install --upgrade pip
```
```bash
python -m pip install -r requirements.txt torch==2.3.0
```
```bash
python -m pip install --no-deps panqec==0.1.6
```

Use one environment route. Do not install dependencies again into an already working environment.

## 3. Validate

```bash
python study.py doctor
```
```bash
python -u study.py check --group all
```
```bash
python study.py list-suites
```
```bash
python study.py list-cases --family all
```

Continue only after validation succeeds.

## 4. Set run options

```bash
export PRESET=v10-main
```
```bash
export PROFILE=matched
```
```bash
export WORKERS=8
```
```bash
export TASK_TIMEOUT=1800
```
```bash
DIST=3
```
```bash
DISTS=3,5,7,15,19
```
```bash
SEED=1879
```
```bash
SEEDS=47,61,83,97,109
```

Edit DIST for single-distance runs and DISTS for distance sweeps. Edit SEED/SEEDS for one/five model-test seeds. Restore these settings after reconnecting. Set WORKERS to fit available CPU and RAM. Use CPU workers; GPUs are not used. Each task uses one CPU thread. TASK_TIMEOUT limits each task in seconds.

Use these preset settings:

| Preset | Distances | Geometry |
|---|---|---|
| v10-main | 3,5,7,15,19 | length2 at d3/d5; length4/8/10 at d7/d15/d19 |
| v10-long-control | 3,5,7,15,19 | length2/3/4/8/10 at d3/d5/d7/d15/d19 |
| v10-d5-short | 5 only | length2 |

Label v10-main distance sweeps as a distance-dependent correlation policy. Use v10-long-control when running the fixed half-code-length rule. Keep both outputs separate. Use the same chosen preset across all cases of a declared study.

Preset budgets:160 training trajectories,100 validation trajectories,300 test trajectories,20 epochs,300 calibration frames/environment. Test trajectories contain14 frames. Each task computes all nine learned methods and physical references. Exact Bayes is available at d3; use the approximate labels above d3.

## 5. Choose a grid

| Suite | Cases | Distances | Seeds | Tasks |
|---|---:|---:|---:|---:|
| hv_pilot | 4 | 1 | 1 | 4 |
| hv_5seed | 4 | 1 | 5 | 20 |
| persistent_5seed | 7 | 1 | 5 | 35 |
| hv_distance_5seed | 4 | 5 | 5 | 100 |
| persistent_distance_5seed | 7 | 5 | 5 | 175 |
| corrected_paper_core | 11 | 5 | 5 | 275 |
| hv_band_pilot | 5 | 1 | 1 | 5 |
| hv_band_5seed | 5 | 1 | 5 | 25 |
| hv_band_distance_5seed | 5 | 5 | 5 | 125 |
| band_distance_5seed | 1 | 5 | 5 | 25 |

Use expanded H+V suites to include the persistent H→V band case alongside the four H+V cases. Use corrected_paper_core for both families without duplicating the band case. Select a grid rather than launching all overlapping grids. Counts change with the number of chosen distances/seeds; five cases with five seeds gives25 tasks.

The launcher plans the grid, runs it, then exports complete tables automatically. It stops before table export if a task fails. Background logs continue after disconnection.

### 4 H+V tasks

```bash
python study.py plan --suite hv_pilot --preset "$PRESET" --profile "$PROFILE" --distances "$DIST" --seeds "$SEED"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh hv_pilot "$DIST" "$SEED" "$PRESET" > logs/hv_4.log 2>&1 &
```
```bash
tail -n 30 logs/hv_4.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite hv_pilot --preset "$PRESET" --profile "$PROFILE"
```

### 20 H+V tasks

```bash
python study.py plan --suite hv_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DIST" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh hv_5seed "$DIST" "$SEEDS" "$PRESET" > logs/hv_20.log 2>&1 &
```
```bash
tail -n 30 logs/hv_20.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite hv_5seed --preset "$PRESET" --profile "$PROFILE"
```

### 35 persistent tasks

```bash
python study.py plan --suite persistent_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DIST" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh persistent_5seed "$DIST" "$SEEDS" "$PRESET" > logs/persistent_35.log 2>&1 &
```
```bash
tail -n 30 logs/persistent_35.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite persistent_5seed --preset "$PRESET" --profile "$PROFILE"
```

### 100 H+V tasks

```bash
python study.py plan --suite hv_distance_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DISTS" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh hv_distance_5seed "$DISTS" "$SEEDS" "$PRESET" > logs/hv_100.log 2>&1 &
```
```bash
tail -n 30 logs/hv_100.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite hv_distance_5seed --preset "$PRESET" --profile "$PROFILE"
```

### 175 persistent tasks

```bash
python study.py plan --suite persistent_distance_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DISTS" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh persistent_distance_5seed "$DISTS" "$SEEDS" "$PRESET" > logs/persistent_175.log 2>&1 &
```
```bash
tail -n 30 logs/persistent_175.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite persistent_distance_5seed --preset "$PRESET" --profile "$PROFILE"
```

### 275 combined tasks

```bash
python study.py plan --suite corrected_paper_core --preset "$PRESET" --profile "$PROFILE" --distances "$DISTS" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh corrected_paper_core "$DISTS" "$SEEDS" "$PRESET" > logs/combined_275.log 2>&1 &
```
```bash
tail -n 30 logs/combined_275.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite corrected_paper_core --preset "$PRESET" --profile "$PROFILE"
```

### 5 expanded H+V tasks

```bash
python study.py plan --suite hv_band_pilot --preset "$PRESET" --profile "$PROFILE" --distances "$DIST" --seeds "$SEED"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh hv_band_pilot "$DIST" "$SEED" "$PRESET" > logs/expanded_5.log 2>&1 &
```
```bash
tail -n 30 logs/expanded_5.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite hv_band_pilot --preset "$PRESET" --profile "$PROFILE"
```

### 25 expanded H+V tasks

```bash
python study.py plan --suite hv_band_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DIST" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh hv_band_5seed "$DIST" "$SEEDS" "$PRESET" > logs/expanded_25.log 2>&1 &
```
```bash
tail -n 30 logs/expanded_25.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite hv_band_5seed --preset "$PRESET" --profile "$PROFILE"
```

### 125 expanded H+V tasks

```bash
python study.py plan --suite hv_band_distance_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DISTS" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh hv_band_distance_5seed "$DISTS" "$SEEDS" "$PRESET" > logs/expanded_125.log 2>&1 &
```
```bash
tail -n 30 logs/expanded_125.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite hv_band_distance_5seed --preset "$PRESET" --profile "$PROFILE"
```

### 25 band-only distance tasks

```bash
python study.py plan --suite band_distance_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DISTS" --seeds "$SEEDS"
```
```bash
nohup stdbuf -oL -eL bash run_grid.sh band_distance_5seed "$DISTS" "$SEEDS" "$PRESET" > logs/band_25.log 2>&1 &
```
```bash
tail -n 30 logs/band_25.log
```
After completion, regenerate tables if needed:
```bash
python study.py tables --suite band_distance_5seed --preset "$PRESET" --profile "$PROFILE"
```

## 6. Select custom cases or budgets

List cases:
```bash
python study.py list-cases --family hv_band
```
Run the band case at d5 on the two replication seeds:
```bash
nohup stdbuf -oL -eL python -u study.py run --suite custom --preset v10-main --profile matched --cases band_family_topology --distances 5 --seeds 3109,3253 --workers "$WORKERS" --timeout-seconds "$TASK_TIMEOUT" > logs/custom_band_d5.log 2>&1 &
```
After successful completion:
```bash
python study.py tables --suite custom --preset v10-main --profile matched
```
Pass comma-separated case names with --cases and distance/seed lists with --distances/--seeds. For custom training budgets, append --ntrain, --nval, --ntest, --epochs or --calibration-per-env to both plan and run commands. Set calibration-per-env to a positive multiple of10. Keep the resulting configuration identifier with your paper table.

## 7. Monitor and resume

Inspect the selected log, for example:
```bash
tail -n 40 logs/hv_100.log
```
For continuous monitoring:
```bash
tail -f logs/hv_100.log
```
Press Ctrl-C to stop monitoring; the background study continues.

Check accepted task counts using exactly the launched grid:
```bash
python study.py status --suite hv_distance_5seed --preset "$PRESET" --profile "$PROFILE" --distances "$DISTS" --seeds "$SEEDS"
```
After a failure/disconnection, restore the variables and rerun the identical launcher command. Accepted completed tasks in the same suite/configuration are skipped. Keep source files unchanged during a study. Different suite names store separate task receipts. After reconnecting, use logs/status; shell wait cannot recover a process owned by an earlier shell.

Require `finished: ok=... failed=0` and `RUN AND TABLE EXPORT COMPLETED` in launcher logs. Do not use --allow-partial for paper tables. Tables can report separate configuration directories if several budgets/profiles were launched; use the directory for the intended configuration.

## 8. Open exported tables

Copy the actual directory printed after Tables: into TABLEDIR:
```bash
TABLEDIR='results/SUITE/PRESET/PROFILE/CONFIG_ID/tables/GRID_ID'
```
```bash
cat "$TABLEDIR/MAIN.md"
```
```bash
cat "$TABLEDIR/EXTENDED.md"
```
```bash
head -n 20 "$TABLEDIR/ALL_METHODS.csv"
```
```bash
head -n 20 "$TABLEDIR/LAST_ITERATE_CONTROLS.csv"
```
```bash
head -n 20 "$TABLEDIR/PAIRED.csv"
```
```bash
head -n 20 "$TABLEDIR/DISTANCE_MEANS.csv"
```

Keep MAIN, EXTENDED and ALL_METHODS together to retain all columns. Use LAST_ITERATE_CONTROLS for nonaveraged output comparisons. Retain unfavorable rows and report seed variation. Run the planned five-seed study before making claims across training seeds.
