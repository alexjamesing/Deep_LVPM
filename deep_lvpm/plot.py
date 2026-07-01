import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
from matplotlib import cm


def _to_numpy(x):
    """Best-effort conversion of common tensor types to a NumPy array.

    Supports:
    - NumPy arrays (returned unchanged)
    - PyTorch tensors (via .detach().cpu().numpy())
    Falls back to np.asarray for other array-likes.
    """
    if isinstance(x, np.ndarray):
        return x

    # PyTorch tensor support
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                pass
    except Exception:
        # Torch not installed or x is not a Torch tensor
        pass

    # Generic fallback
    return np.asarray(x)


def _get_colormap(colormap_name):
    """Return a Matplotlib colormap across old and new Matplotlib versions."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps.get_cmap(colormap_name)
    return cm.get_cmap(colormap_name)


# def plot_correlation_chord_gradient(
#     corr_matrix,
#     labels,
#     min_corr=0.2,
#     figsize=(8, 8),
#     node_cmap_name="Pastel1",   # pastel, seaborn-like
#     save_path=None,
#     dpi=300,
#     show=True,
# ):
#     """
#     Plot a chord diagram for a correlation matrix with:
#       * one color per variable (node), in a pastel gradient
#       * chord thickness clearly proportional to correlation strength
#       * chord color smoothly transitioning between the two node colors

#     Parameters
#     ----------
#     corr_matrix : array-like, shape (n, n)
#         Symmetric correlation matrix with values in [-1, 1].
#     labels : list of str, length n
#         Labels for rows/columns of the matrix.
#     min_corr : float
#         Absolute correlation threshold; smaller values are not drawn.
#     figsize : tuple
#         Figure size.
#     node_cmap_name : str
#         Name of the matplotlib colormap used to generate node colors.
#     save_path : str or None
#         If given, save the figure to this path (e.g. "out/chord.png").
#         If None, the figure is not saved.
#     dpi : int
#         Resolution (dots per inch) when saving.
#     show : bool
#         If True, display the plot with plt.show().

#     Returns
#     -------
#     fig, ax : matplotlib Figure and Axes
#     """
#     corr = np.asarray(corr_matrix)
#     if corr.shape[0] != corr.shape[1]:
#         raise ValueError("corr_matrix must be square.")
#     n = corr.shape[0]
#     if len(labels) != n:
#         raise ValueError("labels length must match corr_matrix size.")

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     # Angles of node centers
#     angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
#     base_radius = 1.0
#     inner_radius = 0.9
#     outer_radius = 1.1

#     # Pastel colors for nodes, sampled along the colormap as a smooth gradient
#     cmap_nodes = cm.get_cmap(node_cmap_name)
#     node_colors = [cmap_nodes(i / max(n - 1, 1)) for i in range(n)]

#     # Draw node arcs
#     for i, angle in enumerate(angles):
#         theta1 = np.degrees(angle - np.pi / n)
#         theta2 = np.degrees(angle + np.pi / n)
#         wedge = Wedge(
#             center=(0, 0),
#             r=outer_radius,
#             theta1=theta1,
#             theta2=theta2,
#             width=outer_radius - inner_radius,
#             facecolor=node_colors[i],
#             edgecolor="white",
#             linewidth=1.0,
#         )
#         ax.add_patch(wedge)

#     # Place labels slightly outside the ring
#     for i, angle in enumerate(angles):
#         label_angle = angle
#         x = (outer_radius + 0.18) * np.cos(label_angle)
#         y = (outer_radius + 0.18) * np.sin(label_angle)

#         ha = "left" if x >= 0 else "right"
#         rotation = np.degrees(label_angle)
#         if x < 0:
#             rotation += 180  # keep text upright

#         ax.text(
#             x,
#             y,
#             labels[i],
#             ha=ha,
#             va="center",
#             rotation=rotation,
#             rotation_mode="anchor",
#             fontsize=11,
#         )

#     # Precompute node positions
#     node_xy = np.column_stack(
#         (base_radius * np.cos(angles), base_radius * np.sin(angles))
#     )

#     def add_chord(p0, p1, color0, color1, corr_val, min_corr,
#                   lw_min=1.5, lw_max=9.0):
#         """
#         Draw a single chord as a quadratic Bézier curve with gradient color.
#         Line width is a *linear* rescaling of |corr_val| from [min_corr, 1]
#         into [lw_min, lw_max], so strength is visually obvious.
#         """
#         num_points = 80
#         ts = np.linspace(0, 1, num_points)

#         # Control point towards the center (you can tweak 0.3 for curvature)
#         control = 0.3 * (p0 + p1) / 2.0

#         points = np.empty((num_points, 2))
#         colors = np.empty((num_points, 4))
#         c0 = np.array(color0)
#         c1 = np.array(color1)

#         for k, t in enumerate(ts):
#             # Quadratic Bézier interpolation
#             points[k] = (
#                 (1 - t) ** 2 * p0
#                 + 2 * (1 - t) * t * control
#                 + t**2 * p1
#             )
#             colors[k] = (1 - t) * c0 + t * c1

#         segments = np.stack([points[:-1], points[1:]], axis=1)
#         seg_colors = colors[:-1]

#         # Strength-normalised width and alpha
#         abs_val = abs(corr_val)
#         strength = (abs_val - min_corr) / (1.0 - min_corr)
#         strength = max(0.0, min(1.0, strength))
#         lw = lw_min + strength * (lw_max - lw_min)
#         alpha = 0.35 + 0.55 * strength  # stronger = more opaque

#         lc = LineCollection(segments, colors=seg_colors,
#                             linewidths=lw, alpha=alpha)
#         ax.add_collection(lc)

#     # Draw chords
#     for i in range(n):
#         for j in range(i + 1, n):
#             val = corr[i, j]
#             if np.isnan(val) or abs(val) < min_corr:
#                 continue
#             add_chord(node_xy[i], node_xy[j],
#                       node_colors[i], node_colors[j],
#                       val, min_corr)

#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.5, 1.5)

#     # Save if requested
#     if save_path is not None:
#         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

#     # Show if requested
#     if show:
#         plt.show()

#     return fig, ax


# # ------------------------------------------------------------------
# # Example data: synthetic multi-omics correlation matrix
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     omics_labels = [
#         "Genomics",
#         "Transcriptomics",
#         "Proteomics",
#         "Metabolomics",
#         "Epigenomics",
#         "Microbiome",
#         "Phosphoproteomics",
#     ]

#     n = len(omics_labels)
#     rng = np.random.default_rng(0)

#     # Start from identity
#     base = np.eye(n)

#     # Strong positive correlations along a plausible omics chain
#     strong_pairs = [
#         (0, 1),  # Genomics - Transcriptomics
#         (1, 2),  # Transcriptomics - Proteomics
#         (2, 3),  # Proteomics - Metabolomics
#         (3, 4),  # Metabolomics - Epigenomics
#         (4, 0),  # Epigenomics - Genomics
#     ]
#     for i, j in strong_pairs:
#         val = rng.uniform(0.7, 0.95)
#         base[i, j] = base[j, i] = val

#     # Moderate correlations
#     moderate_pairs = [
#         (0, 2),
#         (1, 3),
#         (2, 4),
#         (3, 5),
#         (4, 6),
#         (1, 6),
#     ]
#     for i, j in moderate_pairs:
#         val = rng.uniform(0.3, 0.6)
#         base[i, j] = base[j, i] = val

#     # Add small symmetric noise to other entries
#     noise = rng.normal(0, 0.08, size=(n, n))
#     noise = (noise + noise.T) / 2
#     np.fill_diagonal(noise, 0.0)
#     corr_example = base + noise
#     corr_example = np.clip(corr_example, -1, 1)

#     # Example 1: just display
#     fig, ax = plot_correlation_chord_gradient(
#         corr_example,
#         omics_labels,
#         min_corr=0.25,      # only draw reasonably strong links
#         figsize=(8, 8),
#         node_cmap_name="Pastel1",   # pastel, seaborn-ish
#         save_path=None,             # no saving
#         show=True,                  # display the figure
#     )

#     # Example 2 (commented out): save instead of / as well as display

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
from matplotlib import cm


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
from matplotlib import cm


def plot_correlation_chord_row(
    corr_matrices,
    labels,
    min_corr=0.2,
    first_n_dims=None,
    panel_size=(4, 4),
    max_cols=8,
    node_cmap_name="Pastel1",   # pastel, seaborn-like
    figure_title=None,
    show_edge_labels=False,
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Plot a grid of chord diagrams, one for each correlation matrix.

    Parameters
    ----------
    corr_matrices : list of array-like, each shape (n, n)
        List of symmetric correlation matrices with values in [-1, 1].
        All matrices must have the same dimension n.
    labels : list of str, length n
        Labels for rows/columns of the matrices.
    min_corr : float
        Absolute correlation threshold; smaller values are not drawn.
    panel_size : tuple
        Size of each individual chord plot (width, height).
    max_cols : int
        Maximum number of panels to place in a single row.
    node_cmap_name : str
        Name of the matplotlib colormap used to generate node colors.
    figure_title : str or None
        If provided, used as a centered title above the combined row of plots.
    show_edge_labels : bool
        If True, display correlation values (strengths) at the midpoint of each edge.
    save_path : str or None
        If given, save the combined figure to this path (e.g. "out/chords.png").
    dpi : int
        Resolution (dots per inch) when saving.
    show : bool
        If True, display the combined figure with plt.show().

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    # Ensure we have a list
    if not isinstance(corr_matrices, (list, tuple)):
        raise TypeError("corr_matrices must be a list (or tuple) of matrices.")

    # Convert and validate matrices
    # Convert any framework tensors to NumPy for plotting
    corr_list = [_to_numpy(c) for c in corr_matrices]
    if first_n_dims is not None:
        corr_list = corr_list[: int(first_n_dims)]
    if len(corr_list) == 0:
        raise ValueError("corr_matrices list is empty.")

    n = corr_list[0].shape[0]
    for idx, c in enumerate(corr_list):
        if c.shape[0] != c.shape[1]:
            raise ValueError(f"Matrix {idx} is not square.")
        if c.shape[0] != n:
            raise ValueError("All matrices must have the same dimension.")
    if len(labels) != n:
        raise ValueError("labels length must match matrix dimension.")

    n_panels = len(corr_list)
    max_cols = max(1, int(max_cols))
    n_cols = min(n_panels, max_cols)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig_width = panel_size[0] * n_cols
    fig_height = panel_size[1] * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    # Shared geometry for all chord plots
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_radius = 1.0
    inner_radius = 0.9
    outer_radius = 1.1

    # Pastel colors for nodes, sampled along the colormap as a smooth gradient
    cmap_nodes = _get_colormap(node_cmap_name)
    node_colors = [cmap_nodes(i / max(n - 1, 1)) for i in range(n)]

    # Precompute node positions
    node_xy = np.column_stack(
        (base_radius * np.cos(angles), base_radius * np.sin(angles))
    )

    def add_chord(ax, p0, p1, color0, color1, corr_val, min_corr,
                  lw_min=1.5, lw_max=9.0):
        """
        Draw a single chord as a quadratic Bézier curve with gradient color.
        Line width is a linear rescaling of |corr_val| from [min_corr, 1]
        into [lw_min, lw_max], so strength is visually obvious.

        Returns
        -------
        midpoint : np.ndarray, shape (2,)
            Approximate midpoint (x, y) of the chord curve.
        """
        num_points = 80
        ts = np.linspace(0, 1, num_points)

        # Control point towards the center (0.3 controls curvature)
        control = 0.3 * (p0 + p1) / 2.0

        points = np.empty((num_points, 2))
        colors = np.empty((num_points, 4))
        c0 = np.array(color0)
        c1 = np.array(color1)

        for k, t in enumerate(ts):
            # Quadratic Bézier interpolation
            points[k] = (
                (1 - t) ** 2 * p0
                + 2 * (1 - t) * t * control
                + t**2 * p1
            )
            colors[k] = (1 - t) * c0 + t * c1

        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = colors[:-1]

        # Strength-normalised width and alpha
        abs_val = abs(corr_val)
        strength = (abs_val - min_corr) / (1.0 - min_corr)
        strength = max(0.0, min(1.0, strength))
        lw = lw_min + strength * (lw_max - lw_min)
        alpha = 0.35 + 0.55 * strength  # stronger = more opaque

        lc = LineCollection(segments, colors=seg_colors,
                            linewidths=lw, alpha=alpha)
        ax.add_collection(lc)

        # Midpoint of the chord curve (approximate)
        mid_t = 0.5
        midpoint = (
            (1 - mid_t) ** 2 * p0
            + 2 * (1 - mid_t) * mid_t * control
            + mid_t**2 * p1
        )
        return midpoint

    # Draw each chord plot in its own axis
    flat_axes = axes.reshape(-1)
    for idx, (corr, ax) in enumerate(zip(corr_list, flat_axes)):
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw node arcs
        for i, angle in enumerate(angles):
            theta1 = np.degrees(angle - np.pi / n)
            theta2 = np.degrees(angle + np.pi / n)
            wedge = Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=theta1,
                theta2=theta2,
                width=outer_radius - inner_radius,
                facecolor=node_colors[i],
                edgecolor="white",
                linewidth=1.0,
            )
            ax.add_patch(wedge)

        # Place labels slightly outside the ring
        for i, angle in enumerate(angles):
            label_angle = angle
            x = (outer_radius + 0.18) * np.cos(label_angle)
            y = (outer_radius + 0.18) * np.sin(label_angle)

            ha = "left" if x >= 0 else "right"
            rotation = np.degrees(label_angle)
            if x < 0:
                rotation += 180  # keep text upright

            ax.text(
                x,
                y,
                labels[i],
                ha=ha,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=10,
            )

        # Draw chords for this matrix
        for i in range(n):
            for j in range(i + 1, n):
                val = corr[i, j]
                if np.isnan(val) or abs(val) < min_corr:
                    continue
                mid_xy = add_chord(
                    ax,
                    node_xy[i], node_xy[j],
                    node_colors[i], node_colors[j],
                    val, min_corr
                )

                if show_edge_labels:
                    ax.text(
                        mid_xy[0],
                        mid_xy[1],
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Per-panel title like DLV1, DLV2, ...
        ax.set_title(f"DLV{idx + 1}", fontsize=14, pad=20)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    # Add a global, centered title if requested
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=16, y=0.98)

    # Leave room at the top for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig, axes
