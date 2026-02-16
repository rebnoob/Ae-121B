#!/usr/bin/env python3
"""Plot nozzle profiles for equilibrium vs frozen flow from CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot T, P, M vs A/A* for nozzle flow")
    p.add_argument("--eq", default="equilibrium_nozzle.csv", help="Equilibrium CSV path")
    p.add_argument("--fr", default="frozen_nozzle.csv", help="Frozen CSV path")
    p.add_argument(
        "--ar-target",
        type=float,
        default=77.52,
        help="Target nozzle area ratio A_e/A* (used to clip plotted range)",
    )
    p.add_argument(
        "--out",
        default="nozzle_profiles_overlay.png",
        help="Output plot file (.png/.pdf/etc)",
    )
    return p.parse_args()


def split_branches(csv_path: str, ar_target: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    d = df[df["A_over_Astar"].notna()].copy()
    d = d[(d["A_over_Astar"] >= 1.0) & (d["A_over_Astar"] <= ar_target * 1.001)]
    if d.empty:
        raise RuntimeError(f"No valid nozzle points found in {csv_path}")

    throat_idx = d["A_over_Astar"].idxmin()
    subsonic = d.loc[:throat_idx].copy()
    supersonic = d.loc[throat_idx:].copy()

    return subsonic, supersonic


def main() -> None:
    args = parse_args()

    eq_sub, eq_sup = split_branches(args.eq, args.ar_target)
    fr_sub, fr_sup = split_branches(args.fr, args.ar_target)

    c_eq = "#005f73"
    c_fr = "#bb3e03"

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # Temperature
    axes[0].plot(
        eq_sub["A_over_Astar"],
        eq_sub["T_K"],
        "--",
        color=c_eq,
        linewidth=2.0,
        label="Equilibrium (subsonic branch)",
    )
    axes[0].plot(
        eq_sup["A_over_Astar"],
        eq_sup["T_K"],
        "-",
        color=c_eq,
        linewidth=2.2,
        label="Equilibrium (supersonic branch)",
    )
    axes[0].plot(
        fr_sub["A_over_Astar"],
        fr_sub["T_K"],
        "--",
        color=c_fr,
        linewidth=2.0,
        label="Frozen (subsonic branch)",
    )
    axes[0].plot(
        fr_sup["A_over_Astar"],
        fr_sup["T_K"],
        "-",
        color=c_fr,
        linewidth=2.2,
        label="Frozen (supersonic branch)",
    )
    axes[0].set_ylabel("Temperature, T [K]")

    # Pressure
    axes[1].plot(eq_sub["A_over_Astar"], eq_sub["P_Pa"], "--", color=c_eq, linewidth=2.0)
    axes[1].plot(eq_sup["A_over_Astar"], eq_sup["P_Pa"], "-", color=c_eq, linewidth=2.2)
    axes[1].plot(fr_sub["A_over_Astar"], fr_sub["P_Pa"], "--", color=c_fr, linewidth=2.0)
    axes[1].plot(fr_sup["A_over_Astar"], fr_sup["P_Pa"], "-", color=c_fr, linewidth=2.2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Pressure, P [Pa]")

    # Mach number
    axes[2].plot(eq_sub["A_over_Astar"], eq_sub["M"], "--", color=c_eq, linewidth=2.0)
    axes[2].plot(eq_sup["A_over_Astar"], eq_sup["M"], "-", color=c_eq, linewidth=2.2)
    axes[2].plot(fr_sub["A_over_Astar"], fr_sub["M"], "--", color=c_fr, linewidth=2.0)
    axes[2].plot(fr_sup["A_over_Astar"], fr_sup["M"], "-", color=c_fr, linewidth=2.2)
    axes[2].set_ylabel("Mach number, M [-]")
    axes[2].set_xlabel(r"Area ratio, $A/A^*$ [-]")

    for ax in axes:
        ax.axvline(1.0, color="0.35", linewidth=1.2, linestyle=":")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(1.0, args.ar_target)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("LOX/LH2 Nozzle Profiles: Equilibrium vs Frozen", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out).resolve()
    fig.savefig(out, dpi=220)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
