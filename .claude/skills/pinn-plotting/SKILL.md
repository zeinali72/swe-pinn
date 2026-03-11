---
name: pinn-plotting
description: Generate publication-quality figures for PINN flood prediction thesis with strict visual consistency. Use this skill whenever the user asks to create a plot, figure, chart, visualization, or graphic for their thesis or experiment analysis. Also trigger when they say "plot the results", "show me the convergence", "compare the models", "make a figure for Chapter X", "spatial error map", "time-series comparison", or any request involving matplotlib/seaborn output for this project.
---

# PINN Plotting — Publication-Quality Figure Generation

This skill produces visually consistent, thesis-ready figures for eight experiments across three phases of PINN-based urban flood prediction research.

## Setup

Every plotting script must start by applying the global style configuration. Read `references/plot_config.py` and import it at the top of any plotting code:

```python
from plot_config import *
```

If the config file doesn't exist yet, create it at the location the user specifies (default: project root or a `references/` directory).

## Visual Standards

### Typography
- **Font**: Arial everywhere — axis labels, tick labels, titles, legends, annotations, colorbars
- Axis labels: 12pt
- Tick labels: 10pt
- Legend text: 10pt
- Figure titles (if used): 13pt, bold
- Subplot labels (a), (b), (c): 12pt, bold, top-left inside each subplot
- Colorbar labels: 10pt

### Layout
- Single-column figure: `(8, 6)` inches
- Double-column / comparison: `(14, 6)` inches
- Multi-panel grids: scale proportionally
- Line width: 1.5pt for data, 1.0pt for reference/grid
- Grid: light grey `#E0E0E0`, dashed, behind data
- White background, despine top and right spines
- Always call `plt.tight_layout()` before saving
- Always use `bbox_inches='tight'` in `savefig`

### Export
- 300 DPI minimum
- PNG for thesis figures, PDF for journal submission
- Save both formats when the user doesn't specify

## Colour Palette

Two systems — use the correct one for the plot type.

### Primary Scientific Palette
| Name | Hex | Use |
|------|-----|-----|
| Exeter Deep Green | `#003C3C` | Primary model line, PINN predictions |
| Exeter Teal | `#007D69` | Secondary model (DGM) |
| Exeter Mint | `#00C896` | Tertiary model (Fourier-MLP) |
| Blue Heart Navy | `#1B2A4A` | ICM baseline / reference data |
| Blue Heart Ocean | `#2E5C8A` | Observational data |
| Blue Heart Sky | `#7DB4D6` | Confidence intervals, shading |

### Error / Diverging Palette
| Name | Hex | Use |
|------|-----|-----|
| Overprediction | `#C0392B` | Positive error (red) |
| Underprediction | `#2980B9` | Negative error (blue) |
| Zero error | `#F0F0F0` | Neutral centre |

When building a diverging colormap, use `LinearSegmentedColormap.from_list` with `[#2980B9, #F0F0F0, #C0392B]` centred at zero.

## Standard Plot Types

### 1. Convergence Curve
- x = epoch, y = total loss (log scale on y-axis)
- Show PDE, IC, BC, data loss components as separate lines using the primary palette
- Vertical dashed line at the epoch where convergence criterion was met
- Legend in upper-right or best location

### 2. Spatial Error Map
- 2D heatmap of prediction error (h, hu, or hv) at a specific time step
- Diverging red-blue colourmap centred at zero
- Overlay building footprints as black filled polygons (Experiments 2 and 8)
- Overlay domain boundary as black solid line
- Equal aspect ratio (`ax.set_aspect('equal')`)
- Colourbar with units

### 3. Time-Series Comparison
- PINN prediction in Exeter Deep Green, ICM baseline in Blue Heart Navy
- Shaded uncertainty band in Blue Heart Sky if available
- x = time (seconds or formatted), y = variable value with units
- Legend identifying each line

### 4. Scatter Plot (Predicted vs Observed)
- 1:1 reference line in black dashed
- Points coloured by time step using a sequential colourmap
- Text annotation in top-left: R², NSE, RMSE
- Equal axis limits so the 1:1 line is diagonal

### 5. HPO Sensitivity Plot
- Optuna slice plot or parameter importance bar chart
- Primary palette colours
- Annotate the best trial with a marker or label

### 6. Bar Chart (Metric Comparison)
- Grouped bars comparing architectures (MLP, Fourier-MLP, DGM) or experiments
- Exeter Deep Green, Exeter Teal, Exeter Mint for the three architectures
- Error bars where applicable
- Value labels on top of each bar
- Hatching patterns as secondary differentiator if needed for print

### 7. Spatial Snapshot (Flood Extent)
- Sequential blue colourmap: white → Blue Heart Navy for water depth
- Buildings as black filled polygons
- Dry areas (h < threshold) in white
- Equal aspect ratio
- Colourbar labelled "Water depth (m)"

## Implementation Checklist

Before saving any figure, verify:
- [ ] `apply_style()` from plot_config was called (or rcParams set)
- [ ] Correct colour palette used (no matplotlib defaults)
- [ ] Font is Arial (check rcParams)
- [ ] Subplot labels (a), (b), (c) present for multi-panel figures
- [ ] `tight_layout()` called
- [ ] `bbox_inches='tight'` in savefig
- [ ] DPI >= 300
- [ ] Axes labelled with units
- [ ] Legend present and correctly placed
- [ ] For spatial plots: `set_aspect('equal')` and CRS consistency

## Reference Files

- `references/plot_config.py` — Global matplotlib rcParams and colour constants. Read this file to get the exact implementation of `apply_style()` and all colour definitions.
