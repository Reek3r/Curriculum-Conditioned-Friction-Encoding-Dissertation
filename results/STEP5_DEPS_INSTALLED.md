# Pip dependencies installed on step-5 container for probe pipeline

Installed via `python.sh -m pip install --user --break-system-packages` per OPERATING_RULES.md so they live in `~/.local/lib/python3.11/site-packages/` inside the vscode container — these survive container restarts but not container deletion.

| Package | Version | Date | Reason |
|---|---|---|---|
| scikit-learn | 1.8.0 | 2026-05-07 | RidgeCV (linear probe) and MLPRegressor (nonlinear probe) for `probe_rollout_and_fit.py` |
| joblib | 1.5.3 | 2026-05-07 | sklearn dep |
| threadpoolctl | 3.6.0 | 2026-05-07 | sklearn dep |
| matplotlib | 3.10.3 | 2026-05-07 (already present in Isaac Sim core_archive) | `probe_figures.py` |

To replicate on a fresh L40S Brev instance:

```
docker exec vscode bash -c '/workspace/isaaclab/_isaac_sim/python.sh -m pip install --user --break-system-packages scikit-learn'
```

(matplotlib was already bundled with Isaac Sim — no install needed.)

setup_brev.sh was NOT modified per the operating rules — these installs are local to step-5's container.
