#!/usr/bin/env python3
"""Run all causal experiments (C1-C3) sequentially.

C1: causal OT stability under anisotropic deformations.
C2: cost of causal ignorance (Standard vs Causal OT) across 4 datasets.
C3: causal OT on ecoli70 BN data and PCA-conjugacy on real datasets.

Docker default entrypoint: ``python -m fairopt.experiments.run_all``.
"""
import sys
import time
import traceback
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "results" / "causal_run_all.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def run_experiment(name: str, import_path: str, run_fn: str, **kwargs) -> bool:
    log(f"=== STARTING {name} ===")
    start = time.time()
    try:
        __import__(import_path, fromlist=[run_fn])
        mod = sys.modules[import_path]
        runner = getattr(mod, run_fn)
        runner(**kwargs)
        elapsed = time.time() - start
        log(f"=== FINISHED {name} in {elapsed:.1f}s ===")
        return True
    except Exception as e:
        elapsed = time.time() - start
        log(f"=== FAILED {name} after {elapsed:.1f}s: {e} ===")
        traceback.print_exc()
        return False


def main() -> None:
    log("=" * 60)
    log("RUNNING ALL CAUSAL EXPERIMENTS SEQUENTIALLY")
    log("=" * 60)

    experiments = [
        ("C1 Causality Simulation", "fairopt.experiments.c1_causality", "run", {}),
        ("C2 Recourse Causal (CV)", "fairopt.experiments.c2_recourse_causal", "run", {}),
        ("C3 ecoli70 Causal Graph", "fairopt.experiments.c3_causality_real", "run_ecoli70", {}),
        ("C3 Real-Data Conjugacy", "fairopt.experiments.c3_causality_real", "run", {"use_cv": False}),
    ]

    successes = 0
    failures = 0

    for name, import_path, run_fn, kwargs in experiments:
        ok = run_experiment(name, import_path, run_fn, **kwargs)
        if ok:
            successes += 1
        else:
            failures += 1

    log("=" * 60)
    log(f"ALL DONE: {successes} succeeded, {failures} failed")
    log("=" * 60)


if __name__ == "__main__":
    main()
