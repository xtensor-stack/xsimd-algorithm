"""
Minimal matplotlib port of seaborn.relplot (kind="line" only).

Supports only Polars DataFrames. Raises NotImplementedError for unsupported
parameters or types.

AI generated.

Usage
-----
    import polars as pl
    from relplot import relplot

    fig = relplot(
        data=df,
        x="time",
        y="value",
        hue="model",
        palette={"model_a": (0.2, 0.4, 0.8), "model_b": (0.9, 0.3, 0.2)},
        style="variant",
        dashes={"solid_line": (1, 0), "dashed": (5, 5), "dotdash": (3, 5, 1, 5)},
        col="dataset",
        aspect=1.5,
        facet_kws={"sharex": False},
    )
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.lines import Line2D
import polars as pl

type RGB = tuple[float, float, float]
type DashSeq = tuple[int, ...]

_LEGEND_TITLE_FONT = {"weight": "bold", "size": "large"}
_LEGEND_MAX_ENTRIES_PER_ROW = 3


class FacetGrid:
    """Minimal stand-in for seaborn.FacetGrid.

    Exposes the same surface used for post-hoc tweaking:
        g.figure / g.fig   → the matplotlib Figure
        g.axes             → the 2-D ndarray of Axes, shape (n_rows, n_cols)
        g.axes.flat        → flat iterator over every Axes
    """

    __slots__ = ("_fig", "_axes")

    def __init__(self, fig: matplotlib.figure.Figure, axes: np.ndarray) -> None:
        self._fig = fig
        self._axes = axes

    @property
    def figure(self) -> matplotlib.figure.Figure:
        return self._fig

    @property
    def fig(self) -> matplotlib.figure.Figure:
        return self._fig

    @property
    def axes(self) -> np.ndarray:
        return self._axes

    def savefig(self, *args, **kwargs) -> None:
        self._fig.savefig(*args, **kwargs)


def relplot(
    *,
    data: pl.DataFrame,
    x: str,
    y: str,
    kind: str = "line",
    hue: str | None = None,
    palette: dict[str, RGB] | None = None,
    style: str | None = None,
    dashes: dict[str, DashSeq] | None = None,
    row: str | None = None,
    row_order: list[str] | None = None,
    col: str | None = None,
    col_order: list[str] | None = None,
    aspect: float = 1.0,
    height: float = 5.0,
    facet_kws: dict[str, object] | None = None,
) -> FacetGrid:
    """Minimal relplot — line kind only, Polars DataFrames only.

    Parameters
    ----------
    data : pl.DataFrame
    x, y : str
        Column names for the horizontal / vertical axes.
    kind : str
        Only ``"line"`` is implemented.
    hue : str | None
        Column whose distinct values map to colours.  When provided,
        *palette* **must** also be provided as ``dict[str, RGB tuple]``.
    palette : dict[str, tuple[float,float,float]] | None
        Mapping from each *hue* level to an RGB tuple with components in
        [0, 1].  Required when *hue* is set.
    style : str | None
        Column whose distinct values map to dash patterns.  When provided,
        *dashes* **must** also be provided as ``dict[str, tuple[int, ...]]``.
    dashes : dict[str, tuple[int, ...]] | None
        Mapping from each *style* level to a matplotlib on/off ink sequence.
        ``(1, 0)`` or ``()`` → solid line; ``(5, 5)`` → simple dashes;
        ``(3, 5, 1, 5)`` → dash-dot, etc.
    row : str | None
        Column used to facet the data into vertically stacked subplots.
    row_order : list[str] | None
        Order of the *row* levels; levels absent from the data are dropped.
    col : str | None
        Column used to facet the data into side-by-side subplots.
    col_order : list[str] | None
        Order of the *col* levels; levels absent from the data are dropped.
    aspect : float
        Width-to-height ratio of each facet (width = aspect × height).
    height : float
        Height in inches of each facet.
    facet_kws : dict | None
        Currently only ``{"sharex": bool}`` is supported.  *sharey* is
        always ``True``.

    Returns
    -------
    FacetGrid
    """

    # -- Guards --------------------------------------------------------
    if kind != "line":
        raise NotImplementedError(f"Only kind='line' is implemented, got {kind!r}")
    if not isinstance(data, pl.DataFrame):
        raise NotImplementedError(
            f"Only polars.DataFrame is supported, got {type(data).__name__}"
        )
    for name, label in [(x, "x"), (y, "y")]:
        if name not in data.columns:
            raise ValueError(f"{label}={name!r} is not a column in the DataFrame")

    if hue is not None:
        if palette is None:
            raise NotImplementedError(
                "When hue is set, palette must be provided as dict[str, RGB tuple]"
            )
        if not isinstance(palette, dict):
            raise NotImplementedError(
                f"palette must be a dict mapping hue levels to RGB tuples, "
                f"got {type(palette).__name__}"
            )
        if hue not in data.columns:
            raise ValueError(f"hue={hue!r} is not a column in the DataFrame")

    if style is not None:
        if dashes is None:
            raise NotImplementedError(
                "When style is set, dashes must be provided as "
                "dict[str, tuple[int, ...]]"
            )
        if not isinstance(dashes, dict):
            raise NotImplementedError(
                f"dashes must be a dict mapping style levels to dash tuples, "
                f"got {type(dashes).__name__}"
            )
        if style not in data.columns:
            raise ValueError(f"style={style!r} is not a column in the DataFrame")

    if not isinstance(aspect, int | float):
        raise NotImplementedError(
            f"aspect must be a float, got {type(aspect).__name__}"
        )

    sharex: bool = True
    if facet_kws is not None:
        unsupported = set(facet_kws) - {"sharex"}
        if unsupported:
            raise NotImplementedError(
                f"Unsupported facet_kws keys: {unsupported}. "
                f"Only 'sharex' is implemented."
            )
        sharex = facet_kws.get("sharex", True)
        if not isinstance(sharex, bool):
            raise NotImplementedError("facet_kws['sharex'] must be a bool")

    # -- Resolve facet rows / columns ----------------------------------
    row_levels = _facet_levels(data, row, row_order, "row")
    col_levels = _facet_levels(data, col, col_order, "col")
    n_rows = len(row_levels)
    n_cols = len(col_levels)

    # -- Resolve hue / style levels ------------------------------------
    if hue is not None:
        hue_levels = _unique_sorted(data, hue)
        missing = set(hue_levels) - palette.keys()
        if missing:
            raise ValueError(f"palette is missing entries for hue levels: {missing}")
    else:
        hue_levels = [None]

    if style is not None:
        style_levels = _unique_sorted(data, style)
        missing_s = set(style_levels) - dashes.keys()
        if missing_s:
            raise ValueError(f"dashes is missing entries for style levels: {missing_s}")
    else:
        style_levels = [None]

    # -- Create figure + axes ------------------------------------------
    # The plots live in a subfigure so that a suptitle added later on the
    # parent figure is laid out above the legend instead of overlapping it.
    fig = plt.figure(
        figsize=(aspect * height * n_cols, height * n_rows),
        layout="constrained",
        facecolor="none",
    )
    subfig = fig.subfigures()
    axes = subfig.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=sharex,
        sharey=True,
        squeeze=False,
    )

    # -- Plot each facet -----------------------------------------------
    for (row_idx, row_level), (col_idx, col_level) in itertools.product(
        enumerate(row_levels), enumerate(col_levels)
    ):
        ax = axes[row_idx, col_idx]
        ax.set_facecolor("none")

        facet_df = data
        if row is not None:
            facet_df = facet_df.filter(pl.col(row).cast(pl.Utf8) == row_level)
        if col is not None:
            facet_df = facet_df.filter(pl.col(col).cast(pl.Utf8) == col_level)

        for h_level in hue_levels:
            for s_level in style_levels:
                group = facet_df
                if hue is not None:
                    group = group.filter(pl.col(hue).cast(pl.Utf8) == h_level)
                if style is not None:
                    group = group.filter(pl.col(style).cast(pl.Utf8) == s_level)

                if group.is_empty():
                    continue

                group = group.sort(x)

                color = palette[h_level] if hue is not None else None
                linestyle = (
                    _dash_to_linestyle(dashes[s_level]) if style is not None else "-"
                )

                ax.plot(
                    group[x].to_list(),
                    group[y].to_list(),
                    color=color,
                    linestyle=linestyle,
                )

        title_parts = [
            f"{name} = {level}"
            for name, level in ((row, row_level), (col, col_level))
            if name is not None
        ]
        if title_parts:
            ax.set_title(" | ".join(title_parts))

        ax.set_xlabel(x if row_idx == n_rows - 1 else "")
        ax.set_ylabel(y if col_idx == 0 else "")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -- Legend ---------------------------------------------------------
    sections: list[tuple[str, list[Line2D]]] = []
    if hue is not None:
        sections.append(
            (hue, [Line2D([], [], color=palette[lv], label=lv) for lv in hue_levels])
        )
    if style is not None:
        sections.append(
            (
                style,
                [
                    Line2D(
                        [],
                        [],
                        color="grey",
                        linestyle=_dash_to_linestyle(dashes[lv]),
                        label=lv,
                    )
                    for lv in style_levels
                ],
            )
        )

    if sections:
        if row is None and col is None:
            columns = [[_legend_blank(t), *entries] for t, entries in sections]
            handles = [h for column in columns for h in column]
            legend = axes[0, 0].legend(
                handles=handles, loc="upper left", frameon=False
            )
        else:
            rows = _legend_rows(sections, _LEGEND_MAX_ENTRIES_PER_ROW)
            ncol = len(rows[0])
            handles = [row[c] for c in range(ncol) for row in rows]
            legend = subfig.legend(
                handles=handles,
                loc="outside upper center",
                ncol=ncol,
                frameon=False,
            )
        section_titles = {title for title, _ in sections}
        for text in legend.get_texts():
            if text.get_text() in section_titles:
                text.set_fontproperties(_LEGEND_TITLE_FONT)

    return FacetGrid(fig, axes)


# -- Helpers -----------------------------------------------------------


def _legend_blank(label: str = "") -> Line2D:
    return Line2D([], [], linestyle="none", label=label)


def _legend_rows(
    sections: list[tuple[str, list[Line2D]]], max_entries: int
) -> list[list[Line2D]]:
    """Stack sections one below the other, each row led by the section title.

    Rows are padded to the same length since matplotlib fills legends column
    by column.
    """
    rows = []
    for title, entries in sections:
        for i in range(0, len(entries), max_entries):
            header = _legend_blank(title if i == 0 else "")
            rows.append([header, *entries[i : i + max_entries]])
    width = max(len(r) for r in rows)
    return [r + [_legend_blank() for _ in range(width - len(r))] for r in rows]


def _facet_levels(
    df: pl.DataFrame, column: str | None, order: list[str] | None, label: str
) -> list[str | None]:
    if column is None:
        return [None]
    if column not in df.columns:
        raise ValueError(f"{label}={column!r} is not a column in the DataFrame")
    levels = _unique_sorted(df, column)
    if order is None:
        return levels
    available = set(levels)
    return [lv for lv in order if lv in available]


def _unique_sorted(df: pl.DataFrame, column: str) -> list[str]:
    return df.select(pl.col(column).cast(pl.Utf8)).to_series().unique().sort().to_list()


def _dash_to_linestyle(seq: DashSeq) -> str | tuple[int, DashSeq]:
    """Convert an on/off ink tuple to a matplotlib *linestyle* value.

    Seaborn's ``dashes`` dict values are bare ``(on, off, ...)`` tuples.
    Matplotlib expects either a named style or ``(offset, (on, off, ...))``.

    ``()`` and ``(1, 0)`` are treated as solid.
    """
    if not seq or seq == (1, 0):
        return "-"
    return (0, seq)