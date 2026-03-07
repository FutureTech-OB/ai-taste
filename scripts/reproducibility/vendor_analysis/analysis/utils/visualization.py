"""Nature/Science-style plotting utilities for the analysis pipeline.

Provides publication-quality figure helpers including confusion matrices,
grouped bar charts, calibration diagrams, violin plots, heatmaps, and
accuracy curves with confidence bands.  All functions default to the
Nature/Science aesthetic (Arial font, minimal gridlines, no top/right
spines, 300 dpi).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .constants import COLORS, FIGURES_DIR, LABEL_ORDER
from .data_loader import get_project_root


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def set_nature_style() -> None:
    """Set matplotlib rcParams for Nature/Science-quality figures.

    Configures Arial/Helvetica sans-serif font, small font sizes suitable
    for single-column figures, thin axis lines, no top/right spines, and
    300 dpi output.
    """
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,

        # Axes
        'axes.linewidth': 0.5,
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,

        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Legend
        'legend.fontsize': 7,
        'legend.frameon': False,

        # Grid (off by default for Nature style)
        'axes.grid': False,

        # Background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = '',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Blues',
    annotate: bool = True,
    normalize: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a confusion matrix as a coloured heatmap.

    Parameters:
        cm: 2-D confusion matrix array.
        labels: Tick labels for rows (true) and columns (predicted).
        title: Optional figure title.
        ax: Matplotlib axes to draw on.  If ``None``, a new figure is
            created.
        cmap: Colourmap name.
        annotate: Whether to annotate cells with counts/percentages.
        normalize: If ``True``, normalise rows to sum to 1.

    Returns:
        ``(fig, ax)`` tuple.
    """
    set_nature_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
    else:
        fig = ax.figure

    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display = display / row_sums

    im = ax.imshow(display, interpolation='nearest', cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    if annotate:
        thresh = display.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = f'{display[i, j]:.2f}' if normalize else f'{cm[i, j]}'
                ax.text(
                    j, i, text,
                    ha='center', va='center', fontsize=7,
                    color='white' if display[i, j] > thresh else 'black',
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------

def plot_grouped_bar(
    data_dict: Dict[str, List[float]],
    group_labels: List[str],
    title: str = '',
    ylabel: str = '',
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a grouped bar chart comparing methods across categories.

    Parameters:
        data_dict: Mapping from method name to a list of values (one
            per group).
        group_labels: Labels for each group on the x-axis.
        title: Optional figure title.
        ylabel: Y-axis label.
        ax: Matplotlib axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    set_nature_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
    else:
        fig = ax.figure

    method_names = list(data_dict.keys())
    n_methods = len(method_names)
    n_groups = len(group_labels)

    x = np.arange(n_groups)
    width = 0.8 / n_methods

    colour_cycle = list(COLORS.values())

    for idx, name in enumerate(method_names):
        values = data_dict[name]
        offset = (idx - n_methods / 2 + 0.5) * width
        colour = colour_cycle[idx % len(colour_cycle)]
        ax.bar(x + offset, values, width, label=name, color=colour,
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    ax.legend(fontsize=6, loc='best')
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Calibration / reliability diagram
# ---------------------------------------------------------------------------

def plot_calibration_diagram(
    calibration_data: Dict[str, List[Tuple[float, float]]],
    title: str = '',
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a reliability diagram (calibration curve).

    Parameters:
        calibration_data: Mapping from method name to a list of
            ``(mean_predicted_prob, fraction_of_positives)`` tuples
            representing each calibration bin.
        title: Optional figure title.
        ax: Matplotlib axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    set_nature_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
    else:
        fig = ax.figure

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.6, label='Perfectly calibrated')

    colour_cycle = list(COLORS.values())

    for idx, (name, bins) in enumerate(calibration_data.items()):
        if not bins:
            continue
        x_vals = [b[0] for b in bins]
        y_vals = [b[1] for b in bins]
        colour = colour_cycle[idx % len(colour_cycle)]
        ax.plot(x_vals, y_vals, 'o-', color=colour, linewidth=1.0,
                markersize=3, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def plot_violin(
    data_dict: Dict[str, Union[List[float], np.ndarray]],
    title: str = '',
    ylabel: str = '',
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot violin plots comparing distributions.

    Parameters:
        data_dict: Mapping from group name to array of values.
        title: Optional figure title.
        ylabel: Y-axis label.
        ax: Matplotlib axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    set_nature_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
    else:
        fig = ax.figure

    names = list(data_dict.keys())
    data_arrays = [np.asarray(data_dict[n], dtype=float) for n in names]
    positions = list(range(1, len(names) + 1))

    colour_cycle = list(COLORS.values())

    parts = ax.violinplot(data_arrays, positions=positions, showmeans=True,
                          showmedians=True, showextrema=False)

    # Style the violin bodies
    for idx, body in enumerate(parts['bodies']):
        colour = colour_cycle[idx % len(colour_cycle)]
        body.set_facecolor(colour)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)
        body.set_alpha(0.7)

    # Style mean and median lines
    for partname in ('cmeans', 'cmedians'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = '',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'RdYlBu_r',
    annotate: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a general-purpose heatmap.

    Parameters:
        data: 2-D array of values.
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        title: Optional figure title.
        ax: Matplotlib axes.
        cmap: Colourmap name.
        annotate: Whether to annotate cells with values.

    Returns:
        ``(fig, ax)`` tuple.
    """
    set_nature_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
    else:
        fig = ax.figure

    data_arr = np.asarray(data, dtype=float)
    im = ax.imshow(data_arr, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    if annotate:
        thresh = (data_arr.max() + data_arr.min()) / 2.0
        for i in range(data_arr.shape[0]):
            for j in range(data_arr.shape[1]):
                val = data_arr[i, j]
                fmt = f'{val:.2f}' if abs(val) < 100 else f'{val:.0f}'
                ax.text(
                    j, i, fmt,
                    ha='center', va='center', fontsize=6,
                    color='white' if val > thresh else 'black',
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Accuracy curve with CI bands
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    x_values: Union[List[float], np.ndarray],
    y_means: Union[List[float], np.ndarray],
    y_cis: Union[List[Tuple[float, float]], np.ndarray],
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a line with confidence-interval shading.

    Parameters:
        x_values: X-axis values.
        y_means: Mean values for each x.
        y_cis: List of ``(lower, upper)`` tuples for each x, or a 2-D
            array of shape ``(n, 2)``.
        title: Optional figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        ax: Matplotlib axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    set_nature_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
    else:
        fig = ax.figure

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_means, dtype=float)
    ci = np.asarray(y_cis, dtype=float)

    colour = COLORS['ai']

    ax.plot(x, y, '-o', color=colour, linewidth=1.2, markersize=3)
    ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=0.2, color=colour)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Save a figure as both PDF and PNG.

    Parameters:
        fig: Matplotlib figure to save.
        name: Base filename (without extension).
        output_dir: Output directory.  Defaults to
            ``<project_root>/results/figures/``.

    Returns:
        Tuple ``(pdf_path, png_path)``.
    """
    name_path = Path(name)

    # Support either a base filename (e.g., "fig1") or a full/relative path.
    stem = name_path.with_suffix('') if name_path.suffix else name_path

    if stem.is_absolute():
        pdf_path = stem.with_suffix('.pdf')
        png_path = stem.with_suffix('.png')
    else:
        if output_dir is None:
            out = get_project_root() / FIGURES_DIR
        else:
            out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pdf_path = out / stem.with_suffix('.pdf')
        png_path = out / stem.with_suffix('.png')

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(pdf_path), format='pdf', bbox_inches='tight')
    fig.savefig(str(png_path), format='png', dpi=300, bbox_inches='tight')

    return pdf_path, png_path
