"""diag_03_activation_hook.py

Phase 1 diagnostic: confirm that the bare nn.Sequential reconstruction of the
actor accepts a (N, 70) input, that a forward hook on `actor[3]` (the second
ELU) fires once per forward call, and that the captured activation is shape
(N, 64) and finite.

Per METHODOLOGY_LOCKED_SPECS.md the actor layout is:
    actor[0]  Linear(70, 64)
    actor[1]  ELU
    actor[2]  Linear(64, 64)
    actor[3]  ELU             <- hook here, this is Layer 2
    actor[4]  Linear(64, 2)

Usage:
    python.sh diag_03_activation_hook.py <path-to-.pt>

Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
import torch
import torch.nn as nn


def build_actor():
    return nn.Sequential(
        nn.Linear(70, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to model_milestone_*.pt")
    ap.add_argument("--n", type=int, default=8, help="batch size for the forward")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag_03] device: {device}")

    actor = build_actor().to(device)

    print(f"[diag_03] loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    actor_sd = {k.replace("actor.", "", 1): v for k, v in sd.items() if k.startswith("actor.")}
    missing_unexpected = actor.load_state_dict(actor_sd, strict=True)
    print(f"[diag_03] load_state_dict result: {missing_unexpected}")
    actor.eval()

    captured = []

    def hook(_module, _inp, out):
        captured.append(out.detach().clone())

    handle = actor[3].register_forward_hook(hook)

    try:
        torch.manual_seed(0)
        x = torch.randn(args.n, 70, device=device)
        with torch.no_grad():
            y = actor(x)
        print(f"[diag_03] forward output shape: {tuple(y.shape)}")
    finally:
        handle.remove()

    failures = []

    if len(captured) != 1:
        failures.append(f"hook fired {len(captured)} times, expected 1")
    else:
        act = captured[0]
        print(f"[diag_03] activation shape: {tuple(act.shape)}, dtype: {act.dtype}")
        print(f"[diag_03] activation min/max/mean: "
              f"{act.min().item():.4f} / {act.max().item():.4f} / {act.mean().item():.4f}")
        if tuple(act.shape) != (args.n, 64):
            failures.append(f"activation shape {tuple(act.shape)}, expected ({args.n}, 64)")
        if not torch.isfinite(act).all():
            n_nonfinite = (~torch.isfinite(act)).sum().item()
            failures.append(f"activation contains {n_nonfinite} non-finite values")

    if failures:
        print("[diag_03] FAIL:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)

    print("[diag_03] PASS: forward hook fires on actor[3], yields finite (N, 64) activation")
    sys.exit(0)


if __name__ == "__main__":
    main()
