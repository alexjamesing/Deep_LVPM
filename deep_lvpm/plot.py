import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge


def _to_numpy(x):
    """Convert a tensor or array-like to a NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def plot_correlation_chord_row(
    corr_matrices,
    labels,
    min_corr=0.2,
    panel_size=(4, 4),
    node_cmap_name="Pastel1",  # pastel, seaborn-like
    figure_title=None,
    show_edge_labels=False,
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Plot a row of chord diagrams, one for each correlation matrix.

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
        Total figure width is panel_size[0] * number_of_matrices.
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

    # Create figure with one row of subplots
    fig_width = panel_size[0] * n_panels
    # Add a bit of extra vertical space for the global title
    fig_height = panel_size[1] + 0.8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
    if n_panels == 1:
        axes = np.array([axes])

    # Shared geometry for all chord plots
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_radius = 1.0
    inner_radius = 0.9
    outer_radius = 1.1

    # Pastel colors for nodes, sampled along the colormap as a smooth gradient
    cmap_nodes = cm.get_cmap(node_cmap_name)
    node_colors = [cmap_nodes(i / max(n - 1, 1)) for i in range(n)]

    # Precompute node positions
    node_xy = np.column_stack(
        (base_radius * np.cos(angles), base_radius * np.sin(angles))
    )

    def add_chord(
        ax, p0, p1, color0, color1, corr_val, min_corr, lw_min=1.5, lw_max=9.0
    ):
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
            points[k] = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t**2 * p1
            colors[k] = (1 - t) * c0 + t * c1

        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = colors[:-1]

        # Strength-normalised width and alpha
        abs_val = abs(corr_val)
        strength = (abs_val - min_corr) / (1.0 - min_corr)
        strength = max(0.0, min(1.0, strength))
        lw = lw_min + strength * (lw_max - lw_min)
        alpha = 0.35 + 0.55 * strength  # stronger = more opaque

        lc = LineCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha)
        ax.add_collection(lc)

        # Midpoint of the chord curve (approximate)
        mid_t = 0.5
        midpoint = (
            (1 - mid_t) ** 2 * p0 + 2 * (1 - mid_t) * mid_t * control + mid_t**2 * p1
        )
        return midpoint

    # Draw each chord plot in its own axis
    for idx, (corr, ax) in enumerate(zip(corr_list, axes)):
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
                    node_xy[i],
                    node_xy[j],
                    node_colors[i],
                    node_colors[j],
                    val,
                    min_corr,
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
