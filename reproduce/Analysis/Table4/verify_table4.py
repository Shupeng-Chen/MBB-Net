#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

RAW = (
    ROOT
    / "expected_results/Analysis/Table4/"
      "ShapeNet55_complexity.json"
)

REF = (
    ROOT
    / "expected_results/Analysis/Table4/"
      "table4_reference.json"
)


raw = json.loads(
    RAW.read_text(encoding="utf-8")
)

ref = json.loads(
    REF.read_text(encoding="utf-8")
)


by_mode = {
    item["mode"]: item
    for item in raw
}

mapping = {
    "No-Bridge": "no_bridge",
    "Full": "full",
    "MBB": "ours",
}


failed = []


for paper_name, raw_mode in mapping.items():

    r = by_mode[raw_mode]
    e = ref["methods"][paper_name]

    actual = {
        "total_params_M":
            round(r["params_total"] / 1e6, 3),

        "active_params_M":
            round(r["params_active"] / 1e6, 3),

        "flops_G":
            round(r["flops"] / 1e9, 3),

        "latency_ms":
            round(r["latency_mean_ms"], 3),

        # Paper reports peak allocated GPU memory,
        # rounded to the nearest MiB.
        "memory_MB":
            round(r["gpu_peak_allocated_mib"]),
    }

    print()
    print(paper_name)

    for key, expected in e.items():

        got = actual[key]

        same = got == expected

        print(
            f"  {key:18s}"
            f" raw-derived={got}"
            f" expected={expected}"
            f" exact={same}"
        )

        if not same:
            failed.append(
                f"{paper_name}:{key}"
            )


print()
print("failed =", failed)

if failed:
    raise SystemExit(
        "[FAIL] Table 4 reference mismatch."
    )

print(
    "[PASS] TABLE 4 RAW ARTIFACT "
    "REPRODUCES ALL PAPER VALUES."
)
