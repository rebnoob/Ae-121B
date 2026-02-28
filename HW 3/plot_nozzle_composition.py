#!/usr/bin/env python3
"""Create three log-log composition plots for nozzle flow (equilibrium vs frozen)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cantera as ct
except ImportError as exc:
    raise SystemExit("Cantera is required. Install with: pip install cantera") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot species mole fractions vs A/A*")
    p.add_argument("--eq", default="equilibrium_nozzle.csv", help="Equilibrium CSV")
    p.add_argument("--fr", default="frozen_nozzle.csv", help="Frozen CSV")
    p.add_argument("--mech", default="h2o2_highT.cti", help="Mechanism file (.cti or .yaml)")
    p.add_argument("--ar-target", type=float, default=77.52, help="Target A_e/A*")
    p.add_argument(
        "--outdir",
        default=".",
        help="Output directory for plots",
    )
    return p.parse_args()


def cti_to_yaml_if_needed(mech_path: Path) -> Path:
    if mech_path.suffix.lower() != ".cti":
        return mech_path

    out = mech_path.with_suffix(".yaml")
    import cantera.cti2yaml as cti2yaml

    if (not out.exists()) or (out.stat().st_mtime < mech_path.stat().st_mtime):
        cti2yaml.convert(filename=str(mech_path), output_name=str(out))
    return out


def species_mw_map(mech: str) -> Dict[str, float]:
    mech_path = Path(mech).expanduser().resolve()
    if not mech_path.exists():
        raise FileNotFoundError(f"Mechanism not found: {mech_path}")

    load_path = cti_to_yaml_if_needed(mech_path)
    gas = ct.Solution(str(load_path))
    return {
        name: float(gas.molecular_weights[gas.species_index(name)])
        for name in gas.species_names
    }


def add_mole_fractions(df: pd.DataFrame, mw_map: Dict[str, float]) -> pd.DataFrame:
    ycols = [c for c in df.columns if c.startswith("Y_")]
    if not ycols:
        raise RuntimeError("No Y_* columns found in CSV")

    species = [c[2:] for c in ycols]
    missing = [s for s in species if s not in mw_map]
    if missing:
        raise RuntimeError(f"Species missing from mechanism: {missing}")

    inv_mw = np.array([1.0 / mw_map[s] for s in species])
    y = df[ycols].to_numpy(dtype=float)

    n = y * inv_mw[None, :]
    nsum = n.sum(axis=1, keepdims=True)
    x = np.divide(n, nsum, out=np.zeros_like(n), where=nsum > 0.0)

    for i, s in enumerate(species):
        df[f"X_{s}"] = x[:, i]

    return df


def supersonic_branch(df: pd.DataFrame, ar_target: float) -> pd.DataFrame:
    d = df[df["A_over_Astar"].notna()].copy()
    d = d[(d["A_over_Astar"] >= 1.0) & (d["A_over_Astar"] <= ar_target * 1.001)]
    if d.empty:
        raise RuntimeError("No valid A/A* rows to plot")

    i_throat = d["A_over_Astar"].idxmin()
    sup = d.loc[i_throat:].copy()
    sup = sup.sort_values("A_over_Astar")
    return sup


def positive_floor(arr: np.ndarray, floor: float = 1.0e-20) -> np.ndarray:
    return np.clip(arr, floor, 1.0)


def make_single_species_plot(
    out_path: Path,
    eq: pd.DataFrame,
    fr: pd.DataFrame,
    species: str,
    ar_target: float,
) -> None:
    x_eq = eq["A_over_Astar"].to_numpy()
    x_fr = fr["A_over_Astar"].to_numpy()
    y_eq = positive_floor(eq[f"X_{species}"] .to_numpy())
    y_fr = positive_floor(fr[f"X_{species}"] .to_numpy())

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.loglog(x_eq, y_eq, color="#005f73", linewidth=2.3, label="Equilibrium")
    ax.loglog(x_fr, y_fr, color="#bb3e03", linewidth=2.3, linestyle="--", label="Frozen")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(1.0, ar_target)
    ax.set_xlabel(r"Area ratio, $A/A^*$ [-]")
    ax.set_ylabel(f"Mole fraction, $X_{{{species}}}$ [-]")
    ax.set_title(f"{species} Mole Fraction vs Area Ratio")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def make_minor_species_plot(
    out_path: Path,
    eq: pd.DataFrame,
    fr: pd.DataFrame,
    species_list: List[str],
    ar_target: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.3))

    # Color pairs by species; equilibrium solid, frozen dashed.
    cmap = plt.get_cmap("tab10")

    for i, sp in enumerate(species_list):
        color = cmap(i % 10)
        ax.loglog(
            eq["A_over_Astar"],
            positive_floor(eq[f"X_{sp}"].to_numpy()),
            color=color,
            linewidth=2.0,
            label=f"{sp} (eq)",
        )
        ax.loglog(
            fr["A_over_Astar"],
            positive_floor(fr[f"X_{sp}"].to_numpy()),
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{sp} (fr)",
        )

    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(1.0, ar_target)
    ax.set_xlabel(r"Area ratio, $A/A^*$ [-]")
    ax.set_ylabel("Mole fraction, X [-]")
    ax.set_title("Minor Species Mole Fractions vs Area Ratio")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mw_map = species_mw_map(args.mech)

    df_eq = add_mole_fractions(pd.read_csv(args.eq), mw_map)
    df_fr = add_mole_fractions(pd.read_csv(args.fr), mw_map)

    # Use the monotonic nozzle branch from throat to exit.
    eq = supersonic_branch(df_eq, args.ar_target)
    fr = supersonic_branch(df_fr, args.ar_target)

    make_single_species_plot(
        outdir / "composition_h2o_loglog.png", eq, fr, "H2O", args.ar_target
    )
    make_single_species_plot(
        outdir / "composition_h2_loglog.png", eq, fr, "H2", args.ar_target
    )
    make_minor_species_plot(
        outdir / "composition_minor_species_loglog.png",
        eq,
        fr,
        ["H", "O", "O2", "OH", "HO2", "H2O2"],
        args.ar_target,
    )

    print("Saved:")
    print(outdir / "composition_h2o_loglog.png")
    print(outdir / "composition_h2_loglog.png")
    print(outdir / "composition_minor_species_loglog.png")


if __name__ == "__main__":
    main()
