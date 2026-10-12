"""The package's 30-color categorical palette."""


def tab30() -> None:
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(
        [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#ee5c42", "#9467bd", "#cd661d",
            "#e377c2", "#ffbb78", "#bcbd22", "#17becf", "#eead0e", "#aec7e8",
            "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
            "#dbdb8d", "#9edae5", "#f4a460", "#ffe4b5", "#b0c4de", "#9932cc",
            "#ee8262", "#228b22", "#ffe4c4", "#b22222", "#cd853f", "#6a5acd",
        ],
        "tab30",
    )
    try:
        mpl.colormaps.register(cmap, name="tab30")
    except ValueError as exc:
        print(f"Colormap 'tab30' already registered: {exc}")
