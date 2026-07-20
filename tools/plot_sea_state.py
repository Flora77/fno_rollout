# -*- coding: utf-8 -*-
"""
Scatter plot of seen / unseen sea states.

x-axis: wind-sea peak period Tp1
y-axis: swell peak period Tp2
color depth: total significant wave height Hs
blue: seen sea states used in train/val/test
gray: randomly reserved unseen sea states

Author: modify as needed
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable


# =============================================================================
# Basic parameters
# =============================================================================
OUT_DIR = Path("./sea_state_scatter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

g = 9.81
rho_wind = 0.35  # rho = Hs1^2 / Hs^2

Tp1_list = np.array([5.0, 5.5, 6.0, 6.5, 7.0])          # wind-sea peak period
Tp2_list = np.array([10.0, 10.5, 11.0, 11.5, 12.0])      # swell peak period
sp1_list = np.array([0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050])

# Six listed seen sea states from your table.
# Hs is recomputed from Tp1 and sp1, so only Tp1, Tp2, sp1 are needed.
selected_seen_cases = [
    ("U1", 5.5, 10.5, 0.020),
    ("U2", 6.0, 11.0, 0.020),
    ("U3", 5.5, 10.5, 0.040),
    ("U4", 6.0, 11.0, 0.050),
    ("U5", 6.5, 11.5, 0.045),
    ("U6", 7.0, 12.0, 0.050),
]

RANDOM_SEED = 2026
N_UNSEEN = 15

# Set False if you want all points exactly at their Tp1/Tp2 coordinates.
USE_JITTER = True
JITTER_SCALE = 0.035


# =============================================================================
# Construct all 175 sea states
# =============================================================================
rows = []
case_id = 0

for Tp1 in Tp1_list:
    for Tp2 in Tp2_list:
        for sp1 in sp1_list:
            case_id += 1

            # Deep-water peak wavelength of wind-sea component
            Lp1 = g * Tp1**2 / (2.0 * np.pi)

            # Wind-sea significant wave height: Hs1 = sp1 * Lp1
            Hs1 = sp1 * Lp1

            # Total significant wave height:
            # rho = Hs1^2 / Hs^2  =>  Hs = Hs1 / sqrt(rho)
            Hs_total = Hs1 / np.sqrt(rho_wind)

            rows.append({
                "case_id": case_id,
                "Tp1": Tp1,
                "Tp2": Tp2,
                "sp1": sp1,
                "Lp1": Lp1,
                "Hs1": Hs1,
                "Hs": Hs_total,
                "is_selected_seen": False,
                "selected_label": "",
            })

df = pd.DataFrame(rows)

# Mark the six selected seen cases.
for label, Tp1, Tp2, sp1 in selected_seen_cases:
    mask = (
        np.isclose(df["Tp1"], Tp1)
        & np.isclose(df["Tp2"], Tp2)
        & np.isclose(df["sp1"], sp1)
    )
    df.loc[mask, "is_selected_seen"] = True
    df.loc[mask, "selected_label"] = label


# =============================================================================
# Randomly select 15 unseen sea states, excluding the six selected seen cases
# =============================================================================
rng = np.random.default_rng(RANDOM_SEED)

candidate_unseen_idx = df.index[~df["is_selected_seen"]].to_numpy()
unseen_idx = rng.choice(candidate_unseen_idx, size=N_UNSEEN, replace=False)

df["split"] = "seen"
df.loc[unseen_idx, "split"] = "unseen"

# Ensure the six table cases are treated as seen.
df.loc[df["is_selected_seen"], "split"] = "seen"


# =============================================================================
# Add small jitter to avoid overlap at the same Tp1/Tp2 location
# =============================================================================
df["Tp1_plot"] = df["Tp1"]
df["Tp2_plot"] = df["Tp2"]

if USE_JITTER:
    # Deterministic circular jitter according to steepness index.
    sp_to_idx = {sp: i for i, sp in enumerate(sp1_list)}
    angles = np.linspace(0, 2 * np.pi, len(sp1_list), endpoint=False)

    jitter_x = []
    jitter_y = []

    for sp in df["sp1"].to_numpy():
        k = sp_to_idx[sp]
        jitter_x.append(JITTER_SCALE * np.cos(angles[k]))
        jitter_y.append(JITTER_SCALE * np.sin(angles[k]))

    df["Tp1_plot"] = df["Tp1"] + np.array(jitter_x)
    df["Tp2_plot"] = df["Tp2"] + np.array(jitter_y)


# =============================================================================
# Plot
# =============================================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(6.4, 5.2))

norm = Normalize(vmin=df["Hs"].min(), vmax=df["Hs"].max())
cmap_seen = get_cmap("Blues")
cmap_unseen = get_cmap("Greys")

seen = df[df["split"] == "seen"].copy()
unseen = df[df["split"] == "unseen"].copy()

# Convert Hs to color depth.
seen_colors = cmap_seen(0.35 + 0.60 * norm(seen["Hs"].to_numpy()))
unseen_colors = cmap_unseen(0.35 + 0.55 * norm(unseen["Hs"].to_numpy()))

ax.scatter(
    seen["Tp1_plot"],
    seen["Tp2_plot"],
    s=52,
    c=seen_colors,
    marker="o",
    edgecolors="#1f4e8c",
    linewidths=0.45,
    alpha=0.95,
    label="Seen sea states",
    zorder=3,
)

ax.scatter(
    unseen["Tp1_plot"],
    unseen["Tp2_plot"],
    s=58,
    c=unseen_colors,
    marker="o",
    edgecolors="black",
    linewidths=0.55,
    alpha=0.95,
    label="Unseen sea states",
    zorder=4,
)

# Highlight and label the six table cases.
selected = df[df["is_selected_seen"]].copy()
ax.scatter(
    selected["Tp1_plot"],
    selected["Tp2_plot"],
    s=105,
    facecolors="none",
    edgecolors="red",
    linewidths=1.2,
    zorder=6,
    label="Listed seen cases",
)

for _, row in selected.iterrows():
    ax.text(
        row["Tp1_plot"] + 0.035,
        row["Tp2_plot"] + 0.035,
        row["selected_label"],
        fontsize=9,
        color="red",
        ha="left",
        va="bottom",
        zorder=7,
    )

# Axes
ax.set_xlabel(r"Wind-sea peak period $T_{p1}$ / s")
ax.set_ylabel(r"Swell peak period $T_{p2}$ / s")

ax.set_xlim(Tp1_list.min() - 0.25, Tp1_list.max() + 0.25)
ax.set_ylim(Tp2_list.min() - 0.25, Tp2_list.max() + 0.25)

ax.set_xticks(Tp1_list)
ax.set_yticks(Tp2_list)

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax.legend(loc="upper left", frameon=True)

# Colorbar for Hs depth.
sm = ScalarMappable(norm=norm, cmap=cmap_seen)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.025)
cbar.set_label(r"Total significant wave height $H_s$ / m")

ax.set_title("Distribution of Seen and Unseen Mixed-Sea States")

fig.tight_layout()

out_png = OUT_DIR / "seen_unseen_sea_states_scatter.png"
out_pdf = OUT_DIR / "seen_unseen_sea_states_scatter.pdf"
out_svg = OUT_DIR / "seen_unseen_sea_states_scatter.svg"
out_csv = OUT_DIR / "sea_state_split_175_cases.csv"

fig.savefig(out_png, dpi=1000, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"Saved figure: {out_png}")
print(f"Saved figure: {out_pdf}")
print(f"Saved figure: {out_svg}")
print(f"Saved case table: {out_csv}")

print("\nRandomly selected unseen sea states:")
print(df[df["split"] == "unseen"][["case_id", "Tp1", "Tp2", "sp1", "Hs"]].to_string(index=False))