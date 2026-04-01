import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm


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


def plot_correlation_matrix(
    corr_matrices,
    labels,
    panel_size=(4.5, 4.5),
    figure_title=None,
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Plot a row of annotated heatmaps, one for each correlation matrix.

    Drop-in replacement for the former chord-diagram version.  Every
    correlation value is printed inside its own cell, so labels never
    overlap.  The function signature is unchanged; ``min_corr``,
    ``node_cmap_name``, and ``show_edge_labels`` are accepted for
    backward compatibility but ignored.

    Parameters
    ----------
    corr_matrices : list of array-like, each shape (n, n)
        Symmetric correlation matrices with values in [-1, 1].
        All matrices must share the same dimension *n*.
    labels : list of str, length n
        Row / column labels.
    min_corr : float
        Kept for API compatibility; ignored (all values are shown).
    panel_size : tuple of (float, float)
        (width, height) of each panel.
    node_cmap_name : str
        Kept for API compatibility; ignored.
    figure_title : str or None
        Centered suptitle above the row of panels.
    show_edge_labels : bool
        Kept for API compatibility; annotations are always shown.
    save_path : str or None
        If given, save the figure to this path.
    dpi : int
        Resolution when saving.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    import numpy as np

    # --- validation (unchanged from original) ----------------------------
    if not isinstance(corr_matrices, (list, tuple)):
        raise TypeError("corr_matrices must be a list (or tuple) of matrices.")

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

    # --- figure ----------------------------------------------------------
    fig_width = panel_size[0] * n_panels
    fig_height = panel_size[1] + 0.8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
    if n_panels == 1:
        axes = np.array([axes])

    # --- render each panel -----------------------------------------------
    for idx, (corr, ax) in enumerate(zip(corr_list, axes)):
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Annotate every cell
        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

        ax.set_title(f"DLV{idx + 1}", fontsize=14, pad=10)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # --- global title & layout -------------------------------------------
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=16, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_correlation_graph(
    corr_matrices,
    labels,
    min_corr=0.0,
    panel_size=(4.5, 4.5),
    node_cmap_name="Pastel1",
    figure_title=None,
    show_edge_labels=True,
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Plot a row of network-graph correlation diagrams, one per matrix.

    Each variable is a colored node arranged in a circle; edges connect
    pairs whose absolute correlation exceeds *min_corr*.  Edge width and
    color both encode correlation strength via a sequential colormap,
    matching the style of common multi-omics correlation figures.

    The signature is identical to ``plot_correlation_chord_row`` so the
    two functions are interchangeable.

    Parameters
    ----------
    corr_matrices : list of array-like, each shape (n, n)
        Symmetric correlation matrices with values in [-1, 1].
    labels : list of str, length n
        Node labels.
    min_corr : float
        Absolute-correlation threshold; weaker edges are hidden.
    panel_size : tuple of (float, float)
        (width, height) of each panel.
    node_cmap_name : str
        Colormap used for node colors (one color per variable).
    figure_title : str or None
        Centered suptitle.
    show_edge_labels : bool
        If True, print correlation values on each edge.
    save_path : str or None
        If given, save the figure.
    dpi : int
        Resolution when saving.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """

    # --- validation ------------------------------------------------------
    if not isinstance(corr_matrices, (list, tuple)):
        raise TypeError("corr_matrices must be a list (or tuple) of matrices.")

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

    # --- figure ----------------------------------------------------------
    fig_width = panel_size[0] * n_panels
    fig_height = panel_size[1] + 0.8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height))
    if n_panels == 1:
        axes = np.array([axes])

    # --- shared geometry -------------------------------------------------
    # Circular layout (fixed, same across panels)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    pos = {labels[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

    # Node colors
    cmap_nodes = cm.get_cmap(node_cmap_name)
    node_colors = [cmap_nodes(i / max(n - 1, 1)) for i in range(n)]

    # Edge colormap & normalization (sequential, light→dark)
    edge_cmap = cm.get_cmap("GnBu")
    lw_min, lw_max = 1.5, 12.0

    # --- render each panel -----------------------------------------------
    for idx, (corr, ax) in enumerate(zip(corr_list, axes)):
        ax.set_aspect("equal")
        ax.axis("off")

        G = nx.Graph()
        for i in range(n):
            G.add_node(labels[i])

        edges = []
        weights = []
        for i in range(n):
            for j in range(i + 1, n):
                val = corr[i, j]
                if np.isnan(val) or abs(val) < min_corr:
                    continue
                G.add_edge(labels[i], labels[j], weight=abs(val))
                edges.append((labels[i], labels[j]))
                weights.append(abs(val))

        # Edge drawing
        if weights:
            w_arr = np.array(weights)
            norm = mcolors.Normalize(vmin=min_corr, vmax=1.0)
            edge_colors = [edge_cmap(norm(w)) for w in w_arr]
            edge_widths = lw_min + (w_arr - min_corr) / (1.0 - min_corr) * (
                lw_max - lw_min
            )

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                ax=ax,
                width=edge_widths,
                edge_color=edge_colors,
                alpha=0.8,
            )

        # Nodes
        node_size = 1200
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=labels,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="white",
            linewidths=2.0,
        )

        # Node labels (centered inside nodes)
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=9,
            font_weight="bold",
        )

        # Edge labels — two per edge, one near each node at a fixed
        # distance along the direction toward the other node
        label_dist = 0.25  # distance from node center
        if show_edge_labels and weights:
            for i in range(n):
                for j in range(i + 1, n):
                    val = corr[i, j]
                    if np.isnan(val) or abs(val) < min_corr:
                        continue
                    p0 = np.array(pos[labels[i]])
                    p1 = np.array(pos[labels[j]])
                    d = p1 - p0
                    d_norm = np.linalg.norm(d)
                    if d_norm == 0:
                        continue
                    d_hat = d / d_norm
                    txt = f"{abs(val):.2f}"
                    bbox_props = dict(
                        boxstyle="round,pad=0.1",
                        fc="white",
                        ec="none",
                        alpha=0.7,
                    )
                    # Label near node i
                    ax.text(
                        *(p0 + label_dist * d_hat),
                        txt,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                        bbox=bbox_props,
                    )
                    # Label near node j
                    ax.text(
                        *(p1 - label_dist * d_hat),
                        txt,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                        bbox=bbox_props,
                    )

        ax.set_title(f"DLV{idx + 1}", fontsize=14, pad=20)

    # --- global title & layout -------------------------------------------
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=16, y=0.98)

    fig.tight_layout(rect=[0, 0, 0.93, 0.94])

    # --- weight legend (colorbar) ----------------------------------------
    # Place colorbar in its own axes on the far right, after tight_layout
    sm = cm.ScalarMappable(
        cmap=edge_cmap,
        norm=mcolors.Normalize(vmin=min_corr, vmax=1.0),
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.55])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Weight", fontsize=10)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
