"""
AI generated ipywidget UI for plotting.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import polars as pl
from IPython.display import clear_output, display

import data
import lakeborn


Dash = tuple[float, ...]
Color = tuple[float, float, float]


def build_dashes(inputs: list) -> dict[Any, Dash]:
    """Map sorted inputs to dash patterns (at most 5 levels get one)."""
    dash_styles = [(1, 0), (5, 5), (2, 2), (5, 2, 2, 2), (3, 1, 1, 1, 1, 1)]
    return dict(zip(sorted(inputs), dash_styles))


def build_palette(inputs: list) -> dict[Any, Color]:
    """Map sorted inputs to tab10 colors, or tab20 beyond 10 levels."""
    n = len(inputs)
    cmap = matplotlib.colormaps[f"tab{10 if n <= 10 else 20}"]
    colors = [cmap(i)[:3] for i in range(n)]
    return dict(zip(sorted(inputs), colors))


@dataclass(eq=False)
class _FilterRow:
    column: widgets.Dropdown
    value: widgets.Combobox
    box: widgets.HBox


class RelplotUI:
    """Widget UI that builds a filtered relplot from a polars DataFrame."""

    ROLES = ("style", "hue", "col", "row")
    MAX_SUGGESTIONS = 200
    EXPR = "__polars_expr__"
    FIGURE_WIDTH = 12
    WIDGET_WIDTH_PX = 900
    FACET_ASPECT = 1.6

    def __init__(
        self,
        df: pl.DataFrame,
        columns: Sequence[str] | None = None,
        *,
        x: str | None = None,
        y: str | None = None,
        style: str | None = None,
        hue: str | None = None,
        col: str | None = None,
        row: str | None = None,
        title: str | None = None,
        **relplot_kwargs: Any,
    ):
        self.df = df
        self.columns = list(columns) if columns is not None else list(df.columns)
        self._check_columns_exist()
        self.x, self.y = x, y
        self.title = title
        self.relplot_kwargs = relplot_kwargs
        self._syncing = False
        self._last_state: tuple | None = None
        self._figure: matplotlib.figure.Figure | None = None

        defaults = {"style": style, "hue": hue, "col": col, "row": row}
        self.view_dropdowns = {
            role: self._make_dropdown(role, self._column_options(), defaults[role]) for role in self.ROLES
        }
        self._sync_view_options()
        for dd in self.view_dropdowns.values():
            dd.observe(self._on_view_change, names="value")

        self._filter_rows: list[_FilterRow] = []
        self.filters_box = widgets.VBox()
        self._add_filter_row()

        self.output = widgets.Output()
        self.widget = self._build_layout()
        self._auto_plot()

    def _check_columns_exist(self) -> None:
        missing = set(self.columns) - set(self.df.columns)
        if missing:
            raise ValueError(f"Columns not in DataFrame: {sorted(missing)}")

    def _build_layout(self) -> widgets.VBox:
        header = [] if self.title is None else [widgets.HTML(f"<h1>{html.escape(self.title)}</h1>")]
        return widgets.VBox(
            [
                *header,
                widgets.HTML("<b>View</b>"),
                widgets.HBox(list(self.view_dropdowns.values())),
                widgets.HTML("<b>Filters</b>"),
                self.filters_box,
                self.output,
            ]
        )

    @cached_property
    def n_unique(self) -> dict[str, int]:
        return self.df.select(pl.col(self.columns).n_unique()).row(0, named=True)

    @cached_property
    def unique_values(self) -> dict[str, list[Any]]:
        """Sorted non-null unique values, for columns with at most MAX_SUGGESTIONS of them."""
        return {
            col: self._sorted_if_possible(self.df.get_column(col).drop_nulls().unique()).to_list()
            for col, n in self.n_unique.items()
            if n <= self.MAX_SUGGESTIONS
        }

    @staticmethod
    def _sorted_if_possible(s: pl.Series) -> pl.Series:
        """Sort the series, or return it as-is if its dtype can't be sorted."""
        try:
            return s.sort()
        except Exception:
            return s

    def suggestions(self, col: str) -> list[str]:
        return [str(v) for v in self.unique_values.get(col, [])]

    def _column_options(self, exclude: set[str] = frozenset()) -> list[tuple[str, str | None]]:
        return [("None", None)] + [(c, c) for c in self.columns if c not in exclude]

    def _filter_options(self) -> list[tuple[str, str | None]]:
        return [("None", None), ("Polars expr", self.EXPR)] + self._column_options()[1:]

    @staticmethod
    def _make_dropdown(description: str, options: list, value: str | None = None) -> widgets.Dropdown:
        return widgets.Dropdown(
            description=description,
            options=options,
            value=value,
            style={"description_width": "40px"},
            layout=widgets.Layout(width="200px"),
        )

    @staticmethod
    def _set_options_keep_value(dd: widgets.Dropdown, options: list) -> None:
        """Replace a dropdown's options without losing its current selection."""
        current = dd.value
        dd.options = options
        dd.value = current

    def _on_view_change(self, change) -> None:
        if not self._syncing:
            self._sync_view_options()
            self._auto_plot()

    def _sync_view_options(self) -> None:
        """Remove from each view dropdown the columns selected by the other ones."""
        chosen = {role: dd.value for role, dd in self.view_dropdowns.items()}
        self._syncing = True
        try:
            for role, dd in self.view_dropdowns.items():
                taken = {v for r, v in chosen.items() if r != role and v is not None}
                self._set_options_keep_value(dd, self._column_options(exclude=taken))
        finally:
            self._syncing = False

    def _add_filter_row(self) -> None:
        column = self._make_dropdown("filter", self._filter_options())
        value = widgets.Combobox(ensure_option=False, continuous_update=False)
        value.observe(lambda _: self._auto_plot(), names="value")
        row = _FilterRow(column, value, widgets.HBox([column]))
        column.observe(lambda change, row=row: self._on_filter_change(row, change["new"]), names="value")
        self._filter_rows.append(row)
        self._refresh_filters_box()

    def _is_last(self, row: _FilterRow) -> bool:
        return row is self._filter_rows[-1]

    def _on_filter_change(self, row: _FilterRow, col: str | None) -> None:
        row.value.value = ""
        if col is None:
            self._deactivate_filter_row(row)
        else:
            self._activate_filter_row(row, col)
        self._auto_plot()

    def _configure_value_input(self, value: widgets.Combobox, col: str) -> None:
        """Wide free-text input for a polars expression, autocompleted value input for a column."""
        if col == self.EXPR:
            value.options = []
            value.placeholder = f'pl.col("{self.columns[0]}").str.starts_with("")'
            value.layout.width = "500px"
        else:
            value.options = self.suggestions(col)
            value.placeholder = "value"
            value.layout.width = "200px"

    def _activate_filter_row(self, row: _FilterRow, col: str) -> None:
        """Show the value input for `col` and append a fresh empty row if this was the last one."""
        self._configure_value_input(row.value, col)
        row.box.children = (row.column, row.value)
        if self._is_last(row):
            self._add_filter_row()

    def _deactivate_filter_row(self, row: _FilterRow) -> None:
        """Hide the value input, dropping the row unless it is the trailing empty one."""
        row.box.children = (row.column,)
        if not self._is_last(row):
            self._filter_rows.remove(row)
            self._refresh_filters_box()

    def _refresh_filters_box(self) -> None:
        self.filters_box.children = tuple(r.box for r in self._filter_rows)

    @property
    def view(self) -> dict[str, str]:
        return {role: dd.value for role, dd in self.view_dropdowns.items() if dd.value is not None}

    @property
    def filters(self) -> dict[str, list[str]]:
        """Raw texts of the filled-in filter rows, grouped by column (or EXPR for polars expressions)."""
        out: dict[str, list[str]] = {}
        for r in self._filter_rows:
            text = r.value.value.strip()
            if r.column.value is not None and text:
                out.setdefault(r.column.value, []).append(text)
        return out

    @staticmethod
    def _eval_expr(text: str) -> pl.Expr:
        """Evaluate a user-typed polars expression with `pl` in scope."""
        expr = eval(text, {"pl": pl})
        if not isinstance(expr, pl.Expr):
            raise TypeError(f"{text!r} evaluated to {type(expr).__name__}, not a polars expression")
        return expr

    def _column_filter_expr(self, col: str, texts: list[str]) -> pl.Expr:
        """is_in expression with texts cast to the column dtype, falling back to string comparison."""
        dtype = self.df.schema[col]
        try:
            if dtype.is_integer():
                return pl.col(col).is_in([int(t) for t in texts])
            if dtype.is_float():
                return pl.col(col).is_in([float(t) for t in texts])
            if dtype == pl.Boolean:
                return pl.col(col).is_in([t.lower() in {"true", "1", "yes"} for t in texts])
        except ValueError:
            pass
        return pl.col(col).cast(pl.String).is_in(texts)

    def _filter_exprs(self, key: str, texts: list[str]) -> list[pl.Expr]:
        """Polars expressions are kept separate (AND-ed); column values become one OR-ed is_in."""
        if key == self.EXPR:
            return [self._eval_expr(t) for t in texts]
        return [self._column_filter_expr(key, texts)]

    @property
    def filter_expr(self) -> pl.Expr | None:
        """AND of all filters; None when no filter is set."""
        exprs = [e for key, texts in self.filters.items() for e in self._filter_exprs(key, texts)]
        return reduce(lambda a, b: a & b, exprs) if exprs else None

    @property
    def filtered_df(self) -> pl.DataFrame:
        expr = self.filter_expr
        return self.df if expr is None else self.df.filter(expr)

    def _style_maps(self) -> dict[str, dict]:
        """Palette for the hue column and dashes for the style column, built from full-DataFrame levels."""
        maps = {}
        hue, style = self.view.get("hue"), self.view.get("style")
        if hue in self.unique_values:
            maps["palette"] = build_palette(self.unique_values[hue])
        if style in self.unique_values and self.relplot_kwargs.get("kind") == "line":
            maps["dashes"] = build_dashes(self.unique_values[style])
        return maps

    def relplot_args(self) -> dict[str, Any]:
        """All relplot kwargs except data; constructor kwargs override the computed ones."""
        return {"x": self.x, "y": self.y, **self.view, **self._style_maps(), **self.relplot_kwargs}

    def _plot_data(self) -> pl.DataFrame:
        """Filtered data restricted to the columns the plot uses."""
        used = dict.fromkeys([self.x, self.y, *self.view.values()])
        return self.filtered_df.select(list(used))

    def _state(self) -> tuple[dict, dict]:
        return self.view, self.filters

    def _auto_plot(self) -> None:
        """Replot only if the view or filters changed since the last plot."""
        if self._state() != self._last_state:
            self.plot()

    @classmethod
    def _resize(cls, g: Any) -> None:
        """Fix the figure width and give every facet the target aspect ratio.

        A fixed width keeps text at the same on-screen size once the notebook scales
        the image to fit (ipympl does not scale, so it gets an explicit pixel width).
        The height is corrected after a first layout pass, since titles, labels and
        the legend take part of it.
        """
        fig = g.figure
        if isinstance(fig.canvas, widgets.DOMWidget):
            width = cls.WIDGET_WIDTH_PX / fig.dpi
        else:
            width = cls.FIGURE_WIDTH
        n_rows, n_cols = g.axes.shape
        height = width / n_cols / cls.FACET_ASPECT * n_rows
        fig.set_size_inches(width, height)
        fig.draw_without_rendering()
        box = g.axes[0, 0].get_window_extent()
        missing = (box.width / cls.FACET_ASPECT - box.height) / fig.dpi
        fig.set_size_inches(width, height + missing * n_rows)

    def _show(self, g: Any) -> None:
        """Display the figure explicitly, since plt.show() is unreliable inside widget callbacks.

        With ipympl the figure must stay open to remain interactive, so it is only
        closed when replaced.
        """
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        if isinstance(g.figure.canvas, widgets.DOMWidget):
            self._figure = g.figure
            canvas = g.figure.canvas
            canvas.header_visible = False
            canvas.toolbar_position = "top"
            display(canvas)
        else:
            display(g.figure)
            plt.close(g.figure)

    def plot(self) -> Any:
        self._last_state = self._state()
        with self.output:
            clear_output(wait=True)
            if self.x is None or self.y is None:
                print("Set `x` and `y` in the constructor to plot.")
                return None
            try:
                data = self._plot_data()
            except Exception as e:
                print(f"Filter error: {type(e).__name__}: {e}")
                return None
            if data.is_empty():
                print("No rows left after filtering.")
                return None
            with plt.ioff():
                g = lakeborn.relplot(data=data, **self.relplot_args())
            if self.title is not None:
                g.figure.suptitle(self.title, fontsize="xx-large")
            self._resize(g)
            self._show(g)
            return g

    def _ipython_display_(self) -> None:
        display(self.widget)


@dataclass(eq=False)
class _PatternRow:
    name: widgets.Text
    pattern: widgets.Text
    box: widgets.HBox


class Parameters:
    """Widget UI for a url and named patterns, falling back to the NBL_QUERY_PARAMETERS query string."""

    ENV_VAR = "NBL_QUERY_PARAMETERS"

    def __init__(self, url: str | None = None, patterns: dict[str, str] | None = None):
        query = self._query_defaults()
        if url is None:
            url = next(iter(query.get("url", [])), "")
        if patterns is None:
            patterns = dict(zip(query.get("name", []), query.get("pattern", [])))
        self._callbacks: list[Callable[[], None]] = []
        self.url_text = self._make_text("url", url, "600px")
        self.url_text.observe(lambda _: self._notify(), names="value")
        self._pattern_rows: list[_PatternRow] = []
        self.patterns_box = widgets.VBox()
        self.patterns = patterns
        self._preprocessed: tuple[str, pl.DataFrame] | None = None
        self.widget = widgets.VBox([self.url_text, self.patterns_box])

    @classmethod
    def _query_defaults(cls) -> dict[str, list[str]]:
        return parse_qs(os.environ.get(cls.ENV_VAR, ""))

    @staticmethod
    def _make_text(description: str, value: str, width: str) -> widgets.Text:
        return widgets.Text(
            description=description,
            value=value,
            continuous_update=False,
            style={"description_width": "60px"},
            layout=widgets.Layout(width=width),
        )

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call `callback` whenever the url or the patterns change."""
        self._callbacks.append(callback)

    def _notify(self) -> None:
        for callback in self._callbacks:
            callback()

    def _on_pattern_change(self) -> None:
        self._ensure_trailing_empty_row()
        self._notify()

    def _add_pattern_row(self, name: str = "", pattern: str = "") -> None:
        name_text = self._make_text("name", name, "200px")
        pattern_text = self._make_text("pattern", pattern, "600px")
        remove = widgets.Button(icon="times", layout=widgets.Layout(width="32px"))
        row = _PatternRow(name_text, pattern_text, widgets.HBox([name_text, pattern_text, remove]))
        name_text.observe(lambda _: self._notify(), names="value")
        pattern_text.observe(lambda _: self._on_pattern_change(), names="value")
        remove.on_click(lambda _, row=row: self._remove_pattern_row(row))
        self._pattern_rows.append(row)
        self._refresh_patterns_box()

    def _remove_pattern_row(self, row: _PatternRow) -> None:
        self._pattern_rows.remove(row)
        self._ensure_trailing_empty_row()
        self._refresh_patterns_box()
        self._notify()

    def _ensure_trailing_empty_row(self) -> None:
        if not self._pattern_rows or self._pattern_rows[-1].pattern.value.strip():
            self._add_pattern_row()

    def _refresh_patterns_box(self) -> None:
        self.patterns_box.children = tuple(r.box for r in self._pattern_rows)

    @property
    def url(self) -> str:
        return self.url_text.value.strip()

    @url.setter
    def url(self, value: str) -> None:
        self.url_text.value = value

    @property
    def patterns(self) -> dict[str, str]:
        return {
            r.name.value.strip(): r.pattern.value.strip() for r in self._pattern_rows if r.pattern.value.strip()
        }

    @patterns.setter
    def patterns(self, value: dict[str, str]) -> None:
        self._pattern_rows.clear()
        for name, pattern in value.items():
            self._add_pattern_row(name, pattern)
        self._ensure_trailing_empty_row()
        self._refresh_patterns_box()
        self._notify()

    @property
    def preprocessed_df(self) -> pl.DataFrame:
        """Downloaded only when the url changes."""
        if self._preprocessed is None or self._preprocessed[0] != self.url:
            self._preprocessed = (self.url, data.get_preprocessed_df(self.url))
        return self._preprocessed[1]

    def df(self, name: str) -> pl.DataFrame:
        return data.create_df_categories(self.preprocessed_df, pattern=self.patterns[name])

    def _ipython_display_(self) -> None:
        display(self.widget)


class PatternPlots:
    """Parameters followed by one RelplotUI per pattern, each rebuilt only when its url or pattern changes."""

    def __init__(self, params: dict[str, Any] | None = None, relplot: dict[str, Any] | None = None):
        self.params = Parameters(**(params or {}))
        self.relplot = relplot or {}
        self._plots: dict[str, tuple[tuple[str, str], widgets.Widget]] = {}
        self.plots_box = widgets.VBox()
        self.widget = widgets.VBox([self.params.widget, self.plots_box])
        self.params.on_change(self.refresh)
        self.refresh()

    def _build(self, name: str) -> widgets.Widget:
        """The plot widget, or the error it raised so that one bad pattern doesn't hide the others."""
        try:
            return RelplotUI(self.params.df(name), title=name, **self.relplot).widget
        except Exception as e:
            return widgets.HTML(f"<b>{html.escape(name)}</b>: {html.escape(f'{type(e).__name__}: {e}')}")

    def refresh(self) -> None:
        url = self.params.url
        plots = {}
        for name, pattern in self.params.patterns.items():
            old = self._plots.get(name)
            plots[name] = old if old is not None and old[0] == (url, pattern) else ((url, pattern), self._build(name))
        self._plots = plots
        self.plots_box.children = tuple(w for _, w in plots.values())

    def _ipython_display_(self) -> None:
        display(self.widget)
