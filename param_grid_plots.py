# USAGE: python3 param_grid_plots.py <dataset_id> <snapshot (optional, defaults to -1)>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import sys
from matplotlib.gridspec import GridSpec
from fractions import Fraction
from helpers import plot_prettier_lite

RATIOS = [2/1, 5/3, 3/2, 4/3, 5/4, 6/5, 7/6, 1, 8/7, 11/9, 11/8, 9/7,
          7/5, 8/5, 7/4, 9/8, 10/9, 11/10, 12/11, 13/12, 14/13, 13/11, 13/10]

# Axis handling: one place per column type instead of if/elif blocks scattered through the plot code

def geometric_edges(vals):
    """Bin edges at the geometric mean between consecutive log-spaced values."""
    vals = np.asarray(vals, dtype=float)
    edges = np.empty(len(vals) + 1)
    edges[1:-1] = np.sqrt(vals[:-1] * vals[1:])
    edges[0] = vals[0] ** 2 / edges[1]
    edges[-1] = vals[-1] ** 2 / edges[-2]
    return edges
 
def linear_index_edges(n):
    """Bin edges for a discretized/binned integer-index axis (e.g. log_tau_Omega_bin)."""
    edges = np.empty(n + 1)
    edges[1:-1] = np.arange(n - 1) + 0.5
    edges[0] = -0.5
    edges[-1] = n - 0.5
    return edges

def bin_continuous_axis(df, col, nbins):
    """Discretize a continuous column into nbins equal-width bins, returning the
    binned dataframe, the bin-index column name, and the mean value per bin
    (used for tick labels)."""
    df = df.copy()
    bins = np.linspace(df[col].min(), df[col].max(), nbins)
    bin_col = f"{col}_bin"
    df[bin_col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    bin_labels = df.groupby(bin_col, observed=True)[col].mean().values
    return df, bin_col, bin_labels

AXIS_CONFIGS = {
    "Sigma_1au": dict(scale="log", label=r"$\Sigma_{0}\;(\mathrm{g\,cm^{-2}})$",
                       edges=geometric_edges, tick_step=3, tick_fmt=lambda v: f"{round(v, -1):.0f}"),
    "h_1au": dict(scale="log", label=r"$h_{0}$",
                  edges=geometric_edges, tick_step=3, tick_fmt=lambda v: f"{v:.3f}"),
    "log_K2": dict(scale="linear", label=r"$\log \mathcal{K_2}$",
                   edges=geometric_edges, tick_step=1, tick_fmt=lambda v: f"{v:.2f}"),
}

def _axis_config(col):
    if col not in AXIS_CONFIGS:
        raise KeyError(f"No AXIS_CONFIGS entry for '{col}'. Add one before plotting.")
    return AXIS_CONFIGS[col]

# Grid extraction (was duplicated 3x for grid / libration_grid / Delta_grid)

def get_grid(df, x_col, y_col, value_col, x_vals, y_vals, aggfunc="mean"):
    return (
        df.pivot_table(index=x_col, columns=y_col, values=value_col, aggfunc=aggfunc)
          .reindex(index=x_vals, columns=y_vals)
          .values
    )

def _mode_agg(x):
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan

# Cell text formatting (was a long if/elif chain scattered inline; isolated here as the one place it lives)

def format_cell_value(value, value_col, exp=False, bool_mode=None):
    if bool_mode == "symbol":
        return "\u2713" if value else "\u2717"   # checkmark / cross
    if bool_mode == "fraction":
        return f"{round(value * 100)}%"
    if exp:
        return f"{value:.1e}"
    if "survived" in value_col:
        return f"{value:.2g}"
    if "%" in value_col:
        return f"{round(value)}%"
    if "ratio" in value_col:
        frac = Fraction(value).limit_denominator()
        return f"{frac.numerator}:{frac.denominator}" if value in RATIOS else f"{value:.3f}"
    if "kyr" in value_col or "id" in value_col:
        return str(int(value))
    return f"{value:.2f}"

def is_boolean_col(series):
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(non_null):
        return True
    return len(non_null) > 0 and non_null.isin([True, False, 0, 1]).all()

# Single-panel draw: takes an existing ax, returns the mesh so callers (standalone plot or facet grid) can share one colorbar

def draw_param_panel(ax, outcomes, value_col, x_col="Sigma_1au", y_col="h_1au",
                      nbins=None, cmap="viridis", vmin=None, vmax=None, exp=False,
                      log_cmap=False, show_text=True, show_libration=False,
                      show_Delta=False, black_threshold=0.5, h_cut=None,
                      x_tick_step=None, y_tick_step=None, tick_rotation=0,
                      bool_mode="auto"):
    value_is_bool = is_boolean_col(outcomes[value_col])
    if bool_mode == "auto":
        if value_is_bool:
            n_per_cell = outcomes.groupby([x_col, y_col]).size().max()
            bool_mode = "symbol" if n_per_cell == 1 else "fraction"
        else:
            bool_mode = None

    is_binned = nbins is not None
    if is_binned:
        outcomes, x_col, x_bin_medians = bin_grid_axis(outcomes, x_col, nbins)
        outcomes, y_col, y_bin_medians = bin_grid_axis(outcomes, y_col, nbins)

    x_vals = np.sort(outcomes[x_col].dropna().unique())
    y_vals = np.sort(outcomes[y_col].unique())
    if h_cut and not is_binned:
        y_vals = y_vals[y_vals <= h_cut]
    # h_cut assumes real h_1au values, which no longer exist on the axis
    # once y_col has been remapped to a bin index — skip it in that case
    # rather than silently cutting on bin index instead of aspect ratio.

    if bool_mode == "fraction":
        values = get_grid(outcomes, x_col, y_col, value_col, x_vals, y_vals, aggfunc="mean")
    else:
        values = get_grid(outcomes, x_col, y_col, value_col, x_vals, y_vals, aggfunc=_mode_agg)

    libration_grid = None
    if show_libration and value_col in ("inner_P_ratio", "outer_P_ratio"):
        lib_col = "inner_librates" if value_col == "inner_P_ratio" else "outer_librates"
        libration_grid = get_grid(outcomes, x_col, y_col, lib_col, x_vals, y_vals, aggfunc="mean")

    delta_grid = None
    if show_Delta and value_col in ("inner_P_ratio", "outer_P_ratio"):
        delta_col = "inner_P_Delta (%)" if value_col == "inner_P_ratio" else "outer_P_Delta (%)"
        delta_grid = get_grid(outcomes, x_col, y_col, delta_col, x_vals, y_vals, aggfunc="mean")

    # Edges
    if is_binned:
        x_edges = linear_index_edges(len(x_vals))
        y_edges = linear_index_edges(len(y_vals))
        x_plot_vals = np.arange(len(x_vals))
        y_plot_vals = np.arange(len(y_vals))
    else:
        x_edges = _axis_config(x_col)["edges"](x_vals)
        y_edges = _axis_config(y_col)["edges"](y_vals)
        x_plot_vals = x_vals
        y_plot_vals = y_vals

    # Colormap / norm
    if "survived" in value_col or "embryos_inside" in value_col:
        cmap_obj = mcolors.ListedColormap(["#482173", "#2E6F8E", "#29AF7F"])
        norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap_obj.N)
    elif bool_mode == "symbol":
        cmap_obj = mcolors.ListedColormap(["#440154", "#FDE725"])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_obj.N)
    elif bool_mode == "fraction":
        cmap_obj = plt.get_cmap(cmap).copy()
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        cmap_obj = plt.get_cmap(cmap).copy()
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_cmap else None
    cmap_obj.set_bad("lightgray")

    mesh = ax.pcolormesh(
        x_edges, y_edges, values.T, cmap=cmap_obj, norm=norm, shading="flat",
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
    )

    # Axis labels / ticks
    if is_binned:
        xcfg_full, ycfg_full = _axis_config(x_col.replace("_bin", "")), _axis_config(y_col.replace("_bin", ""))
        ax.set_xlabel(xcfg_full["label"])
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        x_step = x_tick_step if x_tick_step is not None else 3
        y_step = y_tick_step if y_tick_step is not None else 3
        ax.set_xticks(np.arange(len(x_bin_medians))[::x_step])
        ax.set_xticklabels([xcfg_full["tick_fmt"](v) for v in x_bin_medians[::x_step]],
                            rotation=tick_rotation, ha="right" if tick_rotation else "center")
        ax.set_ylabel(ycfg_full["label"])
        ax.set_yticks(np.arange(len(y_bin_medians))[::y_step])
        ax.set_yticklabels([ycfg_full["tick_fmt"](v) for v in y_bin_medians[::y_step]])
    else:
        cfg = _axis_config(x_col)
        ax.set_xlabel(cfg["label"])
        ax.set_xscale(cfg["scale"])
        step = x_tick_step if x_tick_step is not None else cfg["tick_step"]
        ax.set_xticks(x_vals[1::step])
        ax.set_xticklabels([cfg["tick_fmt"](v) for v in x_vals[1::step]],
                            rotation=tick_rotation, ha="right" if tick_rotation else "center")

        ycfg = _axis_config(y_col)
        ax.set_ylabel(ycfg["label"])
        ax.set_yscale(ycfg["scale"])
        step = y_tick_step if y_tick_step is not None else ycfg["tick_step"]
        ax.set_yticks(y_vals[1::step])
        ax.set_yticklabels([ycfg["tick_fmt"](v) for v in y_vals[1::step]])
    ax.minorticks_off()

    # Cell annotations
    if show_text or libration_grid is not None or delta_grid is not None:
        for i, x in enumerate(x_plot_vals):
            for j, y in enumerate(y_plot_vals):
                value = values[i, j]
                if np.isnan(value):
                    continue
                if bool_mode == "symbol":
                    color, stroke_color = ("k", "white") if value else ("white", "black")
                else:
                    color = "k" if value > black_threshold else "white"
                    stroke_color = "white" if value > black_threshold else "black"

                if show_text:
                    txt = ax.text(x, y, format_cell_value(value, value_col, exp, bool_mode),
                                  ha="center", va="center", fontsize=3,
                                  weight="bold", color=color, zorder=5)
                    txt.set_path_effects([path_effects.withStroke(linewidth=0.4, foreground=stroke_color)])

                if libration_grid is not None and libration_grid[i, j]:
                    ax.annotate("L", xy=(x, y), xytext=(0, 1), textcoords="offset points",
                                ha="center", va="top", fontsize=5, color="C3", fontweight="bold")

                if delta_grid is not None:
                    ax.annotate(rf"$\Delta = {delta_grid[i, j]}$%", xy=(x, y), xytext=(0, -5),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=4, color=color, fontweight="bold")

    ax.tick_params(axis="both", direction="inout")
    return mesh, bool_mode

# Standalone single-panel figure (drop-in replacement for the original plot_param_grid_map, same public signature)

def plot_param_grid_map(outcomes, value_col, label, x_col="Sigma_1au", y_col="h_1au",
                         nbins=None, cmap="viridis", vmin=None, vmax=None, exp=False,
                         log_cmap=False, show_text=True, show_libration=False,
                         show_Delta=False, black_threshold=0.5, h_cut=None,
                         bool_mode="auto", facet_col=None, facet_val=None,
                         facet_fmt=lambda v: f"{v:.2g}", title=None,
                         x_tick_step=None, y_tick_step=None):

    if facet_col is not None:
        if facet_val is None:
            raise ValueError("facet_val must be given when facet_col is set")
        available = sorted(outcomes[facet_col].unique())
        outcomes = outcomes[outcomes[facet_col] == facet_val]
        if outcomes.empty:
            raise ValueError(f"No rows with {facet_col} == {facet_val!r}; available values: {available}")

    fig, ax = plt.subplots(figsize=(5, 4))
    mesh, resolved_bool_mode = draw_param_panel(
        ax, outcomes, value_col, x_col, y_col, nbins, cmap, vmin, vmax,
        exp, log_cmap, show_text, show_libration, show_Delta,
        black_threshold, h_cut, x_tick_step=x_tick_step, y_tick_step=y_tick_step,
        bool_mode=bool_mode)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label)

    if resolved_bool_mode == "symbol":
        cbar.set_ticks([0, 1], labels=["False", "True"])
        cbar.ax.minorticks_off()
    elif "ratio" in value_col:
        cbar.set_ticks(RATIOS[:8], labels=['2:1', '5:3', '3:2', '4:3', '5:4', '6:5', '7:6', '1:1'])
        cbar.ax.minorticks_off()
    elif "survived" in value_col:
        cbar.set_ticks([1, 2, 3])
    elif "embryos_inside" in value_col or "resonant_embryos" in value_col:
        cbar.set_ticks([0, 1, 2, 3])
        cbar.ax.minorticks_off()
    cbar.ax.tick_params(direction="inout")

    if title is not None:
        ax.set_title(title, fontsize=10)
    elif facet_col is not None:
        ax.set_title(_facet_title(facet_col, facet_val), fontsize=10)

    plt.tight_layout()
    plt.show()    

# 3D extension: small multiples over facet_col (e.g. m_em), one Sigma_1au x h_1au panel per value, sharing one colorbar

def plot_param_grid_facets(outcomes, value_col, label, facet_col, x_col="Sigma_1au",
                            y_col="h_1au", facet_vals=None, ncols=3, panel_size=4.0,
                            cmap="viridis", vmin=None, vmax=None, log_cmap=False,
                            exp=False, show_text=True, show_libration=False,
                            show_Delta=False, black_threshold=0.5, h_cut=None,
                            facet_fmt=lambda v: f"{v:.2g}",
                            wspace=0.45, hspace=0.55,
                            x_tick_step=None, y_tick_step=None, tick_rotation=45):

    if facet_vals is None:
        facet_vals = np.sort(outcomes[facet_col].unique())

    # Panels are smaller than a standalone plot, so default to fewer ticks
    # than the single-panel config unless the caller overrides explicitly.
    if x_tick_step is None:
        x_tick_step = _axis_config(x_col)["tick_step"] * 2
    if y_tick_step is None:
        y_tick_step = _axis_config(y_col)["tick_step"] * 2

    # Shared color scale across all panels unless caller overrides
    if vmin is None or vmax is None and "survived" not in value_col:
        finite_vals = outcomes[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        vmin = finite_vals.min() if vmin is None else vmin
        vmax = finite_vals.max() if vmax is None else vmax

    nrows = int(np.ceil(len(facet_vals) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
                              squeeze=False)

    mesh = None
    resolved_bool_mode = None
    for idx, fv in enumerate(facet_vals):
        ax = axes[idx // ncols, idx % ncols]
        subset = outcomes[outcomes[facet_col] == fv]
        if subset.empty:
            ax.axis("off")
            continue
        mesh, resolved_bool_mode = draw_param_panel(
            ax, subset, value_col, x_col, y_col, None, cmap, vmin, vmax,
            exp, log_cmap, show_text, show_libration, show_Delta,
            black_threshold, h_cut, x_tick_step=x_tick_step, y_tick_step=y_tick_step,
            tick_rotation=tick_rotation)
        ax.set_title(f"{facet_col} = {facet_fmt(fv)}", fontsize=9, pad=8)

    # Turn off unused axes
    for idx in range(len(facet_vals), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, shrink=0.8)
        cbar.set_label(label)
        if resolved_bool_mode == "symbol":
            cbar.set_ticks([0, 1], labels=["False", "True"])
            cbar.ax.minorticks_off()
        elif "ratio" in value_col:
            cbar.set_ticks(RATIOS[:8], labels=['2:1', '5:3', '3:2', '4:3', '5:4', '6:5', '7:6', '1:1'])
            cbar.ax.minorticks_off()
        elif "survived" in value_col:
            cbar.set_ticks([1, 2, 3])

    # Applied after the colorbar so its layout adjustment doesn't clobber this
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.tight_layout()
    plt.show()

FACET_LABELS = {
    "m_em": r"M_{\rm ptsml}",
}

def _facet_title(facet_col, fv):
    if facet_col in FACET_LABELS:
        # exponent = int(np.log10(fv))
        return rf"${FACET_LABELS[facet_col]}$ = {fv}$\,M_\oplus$"
    return f"{facet_col} = {fv:.2g}"

def bin_grid_axis(df, col, n_bins):
    """Split the unique grid values of col into n_bins contiguous, roughly
    equal-count chunks (robust to log spacing, unlike equal-width binning).
    Returns the dataframe with a new bin-index column, the bin-index column
    name, and the median original value per bin (for tick labels)."""
    df = df.copy()
    vals = np.sort(df[col].unique())
    groups = np.array_split(vals, n_bins)
    value_to_bin = {v: i for i, g in enumerate(groups) for v in g}
    bin_col = f"{col}_bin"
    df[bin_col] = df[col].map(value_to_bin)
    bin_medians = np.array([np.median(g) for g in groups])
    return df, bin_col, bin_medians

def plot_param_grid_multi(outcomes, value_col, label, facet_col="m_em",
                           x_col="Sigma_1au", y_col="h_1au", ncols=3,
                           panel_size=4.0, cmap="viridis", vmin=None, vmax=None,
                           cbar_gap=0.15, cbar_width_ratio=0.05,
                           x_tick_step=None, y_tick_step=None, show_text=True,
                           bad='lightgray', start_tick_index=0, nbins=None,
                           aggfunc="mean", discrete=False):

    """
    One panel per facet_col value (e.g. m_em), each a Sigma_1au x h_1au grid
    of value_col, laid out with zero space between panels so adjacent axes
    share a single border line. Tick labels/axis labels only appear on the
    bottom-most active row of each column and the left-most column, so the
    grid reads as one composite plot rather than nine separate ones.

    Since panels touch, there's no room for a title above each one without
    it overlapping the panel above — so the facet value is placed as an
    inset label in the panel's top-left corner instead of a subplot title.

    If value_col is float-valued, cells get a continuous colormap and a
    shared colorbar. If value_col is boolean, cells are flat green (True) /
    red (False) boxes with a legend instead of a colorbar.
    """
    is_bool = is_boolean_col(outcomes[value_col])

    xcfg, ycfg = _axis_config(x_col), _axis_config(y_col)

    if nbins is not None:
        outcomes, x_col, x_bin_medians = bin_grid_axis(outcomes, x_col, nbins)
        outcomes, y_col, y_bin_medians = bin_grid_axis(outcomes, y_col, nbins)

    facet_vals = np.sort(outcomes[facet_col].unique())
    x_tick_step = x_tick_step or xcfg["tick_step"]
    y_tick_step = y_tick_step or ycfg["tick_step"]

    # Discrete level setup: computed once over the full dataset so the
    # category-to-color mapping (and the N/A slot) is identical across
    # facet panels, not re-derived per panel.
    discrete_levels = None
    has_na = False
    if discrete and not is_bool:
        discrete_levels = np.sort(outcomes[value_col].dropna().unique())
        has_na = outcomes[value_col].isna().any()
        n_levels = len(discrete_levels)
        n_colors = n_levels + (1 if has_na else 0)
        base_cmap = plt.colormaps[cmap]
        color_list = [base_cmap(i / max(n_levels - 1, 1)) for i in range(n_levels)]
        if has_na:
            color_list.append(mcolors.to_rgba(bad))
        cmap_obj_base = mcolors.ListedColormap(color_list)
        norm_base = mcolors.BoundaryNorm(np.arange(-0.5, n_colors, 1), cmap_obj_base.N)

    if not is_bool and discrete_levels is None and (vmin is None or vmax is None):
        finite = outcomes[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        vmin = finite.min() if vmin is None else vmin
        vmax = finite.max() if vmax is None else vmax

    nrows = int(np.ceil(len(facet_vals) / ncols))
    if ncols == 1:
        fig = plt.figure(figsize=(5, 5))
    else:
        fig = plt.figure(figsize=(2 + 2.5 * ncols, 1.5 + 2.5 * nrows))

    outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, cbar_width_ratio], wspace=cbar_gap)
    grid_gs = outer_gs[0].subgridspec(nrows, ncols, wspace=0, hspace=0)

    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(grid_gs[r, c])
    cax = fig.add_subplot(outer_gs[1])

    last_active_row = {}
    for idx in range(len(facet_vals)):
        r, c = divmod(idx, ncols)
        last_active_row[c] = r

    mesh = None
    for idx, fv in enumerate(facet_vals):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        subset = outcomes[outcomes[facet_col] == fv]
        if subset.empty:
            ax.axis("off")
            continue

        if nbins is not None:
            x_vals = np.arange(nbins)
            y_vals = np.arange(nbins)
            values = get_grid(subset, x_col, y_col, value_col, x_vals, y_vals,
                               aggfunc=aggfunc)
            x_edges = linear_index_edges(nbins)
            y_edges = linear_index_edges(nbins)
        else:
            x_vals = np.sort(subset[x_col].unique())
            y_vals = np.sort(subset[y_col].unique())
            values = get_grid(subset, x_col, y_col, value_col, x_vals, y_vals,
                               aggfunc=_mode_agg)
            x_edges, y_edges = xcfg["edges"](x_vals), ycfg["edges"](y_vals)

        # values keeps the real numbers for cell text; color_values is a
        # separate array remapped to category indices purely for pcolormesh
        if is_bool:
            cmap_obj = mcolors.ListedColormap(["#d62728", "#2ca02c"])  # False=red, True=green
            norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_obj.N)
            color_values = values
        elif discrete_levels is not None:
            cmap_obj = cmap_obj_base
            norm = norm_base
            finite_mask = ~np.isnan(values)
            color_values = np.full(values.shape, np.nan)
            color_values[finite_mask] = np.searchsorted(discrete_levels, values[finite_mask])
            if has_na:
                color_values[~finite_mask] = len(discrete_levels)
        else:
            cmap_obj = plt.get_cmap(cmap)
            norm = None
            color_values = values

        cmap_obj.set_bad(bad)

        mesh = ax.pcolormesh(x_edges, y_edges, color_values.T, cmap=cmap_obj, norm=norm,
                              vmin=None if norm else vmin, vmax=None if norm else vmax,
                              shading="flat")

        if show_text and not is_bool:
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    v = values[i, j]
                    if np.isnan(v):
                        continue
                    txt = ax.text(
                        x, y, format_cell_value(v, value_col),
                        ha="center", va="center", fontsize=5,
                        color="black", fontweight="bold", zorder=5,
                    )
                    txt.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground="white")])

        ax.set_xscale(xcfg["scale"] if nbins is None else "linear")
        ax.set_yscale(ycfg["scale"] if nbins is None else "linear")
        ax.set_xticks(x_vals[start_tick_index::x_tick_step])
        ax.set_yticks(y_vals[start_tick_index::y_tick_step])
        ax.minorticks_off()
        ax.tick_params(axis="both", direction="inout", labelsize=8)
        
        # # Rotate the labels using tick_params
        # ax.tick_params(axis='x', labelrotation=45)

        # # Ensure proper alignment so text ends under the tick
        # for label in ax.get_xticklabels():
        #     label.set_horizontalalignment('right')

        is_bottom_edge = (r == last_active_row[c])
        is_left_edge = (c == 0)

        x_labels_src = x_bin_medians if nbins is not None else x_vals
        y_labels_src = y_bin_medians if nbins is not None else y_vals

        if is_bottom_edge:
            ax.set_xticklabels([xcfg["tick_fmt"](v) for v in x_labels_src[start_tick_index::x_tick_step]])
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        if is_left_edge:
            ax.set_yticklabels([ycfg["tick_fmt"](v) for v in y_labels_src[start_tick_index::y_tick_step]])
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.text(0.97, 0.03, _facet_title(facet_col, fv), transform=ax.transAxes,
                 ha="right", va="bottom", fontsize=10, color='w')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
            spine.set_zorder(10)

    fig.supxlabel(r"Surface density at 1 au $(\Sigma_0)$ $\left[\text{g cm}^{-2}\right]$", fontsize=12)
    fig.supylabel("Aspect ratio at 1 au $(h_0)$", fontsize=12)

    for idx in range(len(facet_vals), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    cax.axis("off")
    if is_bool:
        legend_handles = [
            patches.Patch(facecolor="#2ca02c", label="True"),
            patches.Patch(facecolor="#d62728", label="False"),
        ]
        cax.legend(handles=legend_handles, loc="center", frameon=False)
    elif mesh is not None:
        cax.axis("on")
        cax.set_frame_on(True)
        cax.set_xticks([])
        cax.set_yticks([])
        cbar = fig.colorbar(mesh, cax=cax, fraction=1.0)
        cbar.set_label(label)

        if discrete_levels is not None:
            n_colors = len(discrete_levels) + (1 if has_na else 0)
            tick_labels = [format_cell_value(v, value_col) for v in discrete_levels]
            if has_na:
                tick_labels.append("N/A")
            cbar.set_ticks(range(n_colors))
            cbar.set_ticklabels(tick_labels)
            cbar.ax.minorticks_off()
        
        
    plt.show()

if __name__ == "__main__":    
    assert len(sys.argv) == 2 or len(sys.argv) == 3
    dataset_id = sys.argv[1]

    if len(sys.argv) == 2:
        snapshot = -1
    else:
        snapshot = int(sys.argv[2])
    outcomes = pd.read_hdf(f"dfs/outcomes{dataset_id}_{snapshot}.h5", key="df") # Load data

    outcomes = outcomes[outcomes['m_em'] <= 0.1] # this prevents the code from breaking
    outcomes['scattered (%)'] = outcomes["em_surv_rate (%)"] - 100*outcomes["embryos_inside"]/6
    outcomes['accreted (%)'] = 100 - outcomes["em_surv_rate (%)"]

    ncols = 4
    x_tick_step = 3
    y_tick_step = 3

    # plot_param_grid_multi(outcomes, "embryos_inside", "Embryos between two planets", show_text=False, ncols=ncols, x_tick_step=x_tick_step, y_tick_step=y_tick_step)
    # plot_param_grid_multi(outcomes, "em_surv_rate (%)", "Total embryos survived (%)", show_text=False, ncols=ncols, x_tick_step=x_tick_step, y_tick_step=y_tick_step)
    # plot_param_grid_multi(outcomes, "resonant_embryos", "Resonant embryos", show_text=False, ncols=ncols, x_tick_step=x_tick_step, y_tick_step=y_tick_step)
    # plot_param_grid_multi(outcomes, "scattered (%)", "Embryos scattered (%)", show_text=False, ncols=ncols, x_tick_step=x_tick_step, y_tick_step=y_tick_step)
    # plot_param_grid_multi(outcomes, "accreted (%)", "Embryos accreted (%)", show_text=False, ncols=ncols, x_tick_step=x_tick_step, y_tick_step=y_tick_step)
    plot_param_grid_multi(outcomes, "sim_id", "Sim ID", ncols=ncols, x_tick_step=x_tick_step, y_tick_step=y_tick_step)
