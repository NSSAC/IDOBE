"""CLI: fit the synthetic generator on the real IDOBE table and emit a CSV.

Example:
    python synthetic/generate.py \
        --input raw_data/outbreaks_disease_location.csv \
        --output synthetic/output/synthetic_outbreaks.csv \
        --n 9962 --seed 0

With no --n, it produces the same number of outbreaks as the source table.
The output CSV matches the raw_data schema exactly, so the existing stats and
baseline pipelines can consume it unchanged.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from generator import SyntheticOutbreakGenerator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="raw_data/outbreaks_disease_location.csv",
                   help="Path to the real outbreaks CSV.")
    p.add_argument("--output", default="synthetic/output/synthetic_outbreaks.csv",
                   help="Where to write the synthetic CSV.")
    p.add_argument("--n", type=int, default=None,
                   help="Number of synthetic outbreaks (default: match input size).")
    p.add_argument("--components", type=int, default=10,
                   help="Number of FPCA shape components.")
    p.add_argument("--roughness", type=float, default=0.04,
                   help="Week-to-week noise; 0 = smooth curves (see generator).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    n = args.n if args.n is not None else len(df)

    gen = SyntheticOutbreakGenerator(n_components=args.components,
                                     roughness=args.roughness, seed=args.seed)
    gen.fit(df)
    synth = gen.sample(n)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synth.to_csv(args.output, index=False)
    print(f"Wrote {len(synth)} synthetic outbreaks to {args.output}")


if __name__ == "__main__":
    main()
