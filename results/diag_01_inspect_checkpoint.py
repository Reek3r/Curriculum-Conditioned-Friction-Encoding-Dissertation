"""diag_01_inspect_checkpoint.py

Phase 1 diagnostic: confirm that an rsl_rl checkpoint contains an actor with the
locked architecture 70 -> 64 -> 64 -> 2 (ELU activations between linear layers).

Per METHODOLOGY_LOCKED_SPECS.md the actor is `nn.Sequential` with:
    actor.0  Linear(70, 64)
    actor.1  ELU
    actor.2  Linear(64, 64)
    actor.3  ELU                <- probe site (Layer 2 activation)
    actor.4  Linear(64, 2)

ELU activation is parameter-free, so it does not appear in the state_dict; we
verify it by linear-layer shape only and rely on the training-code lock for
the activation type.

Usage:
    python.sh diag_01_inspect_checkpoint.py <path-to-.pt>

Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
import torch


EXPECTED_ACTOR_SHAPES = {
    "actor.0.weight": (64, 70),
    "actor.0.bias": (64,),
    "actor.2.weight": (64, 64),
    "actor.2.bias": (64,),
    "actor.4.weight": (2, 64),
    "actor.4.bias": (2,),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to model_milestone_*.pt")
    args = ap.parse_args()

    print(f"[diag_01] loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    print(f"[diag_01] top-level type: {type(ckpt).__name__}")

    if isinstance(ckpt, dict):
        print(f"[diag_01] top-level keys: {sorted(ckpt.keys())}")
        sd = ckpt.get("model_state_dict", ckpt)
    else:
        print("[diag_01] FAIL: top-level is not a dict")
        sys.exit(1)

    if not isinstance(sd, dict):
        print(f"[diag_01] FAIL: model_state_dict is {type(sd).__name__}, not dict")
        sys.exit(1)

    actor_keys = sorted(k for k in sd.keys() if k.startswith("actor."))
    critic_keys = sorted(k for k in sd.keys() if k.startswith("critic."))
    other_keys = sorted(k for k in sd.keys() if not k.startswith(("actor.", "critic.")))

    print(f"[diag_01] actor keys: {actor_keys}")
    print(f"[diag_01] critic keys: {critic_keys}")
    print(f"[diag_01] other keys: {other_keys}")

    print("[diag_01] actor parameter shapes:")
    for k in actor_keys:
        print(f"    {k}: {tuple(sd[k].shape)}")

    failures = []
    for key, expected_shape in EXPECTED_ACTOR_SHAPES.items():
        if key not in sd:
            failures.append(f"missing key {key}")
            continue
        actual = tuple(sd[key].shape)
        if actual != expected_shape:
            failures.append(f"shape mismatch on {key}: got {actual}, expected {expected_shape}")

    extra_actor_linear_keys = [
        k for k in actor_keys
        if k.endswith(".weight") and k not in EXPECTED_ACTOR_SHAPES
    ]
    if extra_actor_linear_keys:
        failures.append(f"unexpected actor linear-layer keys: {extra_actor_linear_keys}")

    if failures:
        print("[diag_01] FAIL:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)

    print("[diag_01] PASS: actor architecture is 70->64->64->2 (linear shapes verified)")
    sys.exit(0)


if __name__ == "__main__":
    main()
