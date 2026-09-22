#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--result",
        required=True,
    )

    p.add_argument(
        "--reference",
        default="expected_results/shapenet55_pcgrad_reference.json",
    )

    p.add_argument(
        "--atol",
        type=float,
        default=1e-12,
    )

    args = p.parse_args()

    ref = load(args.reference)
    cur = load(args.result)

    print("=" * 84)
    print("SHAPENET-55 PCGRAD FINAL VERIFIER")
    print("=" * 84)

    failures = []
    max_diff = 0.0

    nr = int(ref["num_objects"])
    nc = int(cur["num_objects"])

    print(f"\nnum_objects: ref={nr} cur={nc}")

    if nr != nc:
        failures.append(("num_objects", nr, nc))

    for group in ("cd_l2_x1000", "f1_at_1pct"):
        print(f"\n{group}")

        for key in ("simple", "moderate", "hard", "average"):
            a = float(ref[group][key])
            b = float(cur[group][key])
            d = abs(a - b)

            max_diff = max(max_diff, d)

            exact = d <= args.atol
            display = f"{a:.4f}" == f"{b:.4f}"

            print(
                f"  {key:8s} "
                f"ref={a:.15f} "
                f"cur={b:.15f} "
                f"diff={d:.3e} "
                f"4dp={a:.4f}/{b:.4f} "
                f"exact={'PASS' if exact else 'FAIL'} "
                f"display={'PASS' if display else 'FAIL'}"
            )

            if not exact:
                failures.append((group, key, a, b, d))

    a = float(ref["accuracy_object_mean_logits"])
    b = float(cur["accuracy_object_mean_logits"])
    d = abs(a - b)

    max_diff = max(max_diff, d)

    print("\naccuracy_object_mean_logits")
    print(f"  ref={a:.15f}")
    print(f"  cur={b:.15f}")
    print(f"  diff={d:.3e}")
    print(f"  display={100*a:.2f}%/{100*b:.2f}%")

    if d > args.atol:
        failures.append(
            ("accuracy_object_mean_logits", a, b, d)
        )

    print()
    print("=" * 84)
    print(f"global max diff = {max_diff:.15e}")
    print(f"failures        = {len(failures)}")

    if failures:
        print("\n[FAIL] PCGrad canonical verification failed.")
        for x in failures:
            print(" ", x)
        raise SystemExit(1)

    print()
    print(
        "[PASS] ShapeNet-55 PCGrad released checkpoint "
        "matches canonical results."
    )
    print(
        "[PASS] Paper-displayed PCGrad values are reproduced."
    )


if __name__ == "__main__":
    main()
