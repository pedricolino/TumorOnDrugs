"""Plots for cluster composition."""


def plot_cluster_distribution(adata, x_col="clusters", y_col="sample", cmap="Greys", fig_size=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    counts = adata.obs.groupby([x_col, y_col]).size().unstack(fill_value=0)
    normalized = counts.div(counts.sum(axis=0), axis=1)
    if fig_size is None:
        fig_size = (len(normalized.columns), len(normalized.index))
    plt.figure(figsize=fig_size)
    sns.heatmap(normalized, annot=counts, fmt="d", cmap=cmap)
    plt.title(f"Distribution of {x_col} across {y_col}")
    plt.xlabel(y_col)
    plt.ylabel(x_col)
    plt.yticks(rotation=0, ha="right")
    plt.grid(False)
    plt.gca().add_patch(
        plt.Rectangle(
            (0, 0),
            normalized.shape[1],
            normalized.shape[0],
            fill=False,
            edgecolor="black",
            lw=2,
        )
    )
    plt.tight_layout()
    plt.show()
