"""Plotly figures that explain how the pipeline reached its answer.

Every figure here obeys the same three rules, which is why they read as one
system rather than four separate charts:

1. The colour of a stage is fixed by ``ui.theme`` — violet is always dense,
   cyan is always BM25, amber is always the fused/hybrid path, rose is always
   the fallback. A colour learned on the SVG diagram means the same thing here.
2. Paper and plot backgrounds are transparent so the figures sit on the app's
   own gradient instead of punching grey rectangles into it.
3. No chartjunk: no vertical gridlines where they carry nothing, no legend when
   a single series is plotted, no mode bar (passed via ``PLOTLY_CONFIG``).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .theme import COLORS

# Passed to every st.plotly_chart call. A mode bar on a projector is noise.
PLOTLY_CONFIG: dict[str, Any] = {"displayModeBar": False, "responsive": True}

_FONT = "Be Vietnam Pro, Inter, sans-serif"
_MONO = "JetBrains Mono, monospace"


def _base_layout(fig: go.Figure, *, height: int, title: str = "") -> go.Figure:
    """Apply the shared look: transparent, tight margins, Vietnamese-safe font."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=44 if title else 14, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, size=12, color=COLORS["text_dim"]),
        title=dict(
            text=title,
            font=dict(size=13.5, color=COLORS["muted"], family=_FONT),
            x=0,
            xanchor="left",
            y=0.97,
        ) if title else None,
        hoverlabel=dict(
            bgcolor=COLORS["surface_2"],
            bordercolor=COLORS["border"],
            font=dict(family=_FONT, size=12, color=COLORS["text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _empty(message: str, *, height: int = 220) -> go.Figure:
    """Placeholder figure - used when a trace lacks the data for a chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(family=_FONT, size=12.5, color=COLORS["faint"]),
        x=0.5, y=0.5, xref="paper", yref="paper",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height=height)


def _short(title: str, limit: int = 46) -> str:
    return title if len(title) <= limit else title[: limit - 1] + "…"


# --- 1. rank movement --------------------------------------------------------
def rank_movement_slope(trace: dict[str, Any]) -> go.Figure:
    """Slope chart: Dense rank → BM25 rank → final rank, one line per chunk.

    This is the most explanatory graphic in the app. It shows what RRF actually
    did: which chunk each retriever found, which chunk only *one* of them found,
    and which candidates survived into the final top-k.

    Gaps are joined (``connectgaps=True``) but markers are only drawn where a
    rank exists, so a column with no dot reads as "this retriever never returned
    that chunk" while the line still shows the journey.
    """
    rows = list(trace.get("rank_movement") or [])
    if not rows:
        return _empty("Trace không có dữ liệu rank_movement.", height=380)

    ranks = [
        r for row in rows
        for r in (row.get("dense_rank"), row.get("bm25_rank"), row.get("final_rank"))
        if r is not None
    ]
    max_rank = max(ranks) if ranks else 1

    columns = [
        f'<span style="color:{COLORS["dense"]}"><b>Dense</b></span>',
        f'<span style="color:{COLORS["bm25"]}"><b>BM25</b></span>',
        f'<span style="color:{COLORS["hybrid"]}"><b>Xếp hạng cuối</b></span>',
    ]
    column_colors = [COLORS["dense"], COLORS["bm25"], COLORS["hybrid"]]

    fig = go.Figure()
    legend_seen: set[str] = set()

    for row in rows:
        y = [row.get("dense_rank"), row.get("bm25_rank"), row.get("final_rank")]
        kept = row.get("final_rank") is not None
        group = "kept" if kept else "dropped"
        line_color = COLORS["hybrid"] if kept else COLORS["muted"]
        marker_colors = column_colors if kept else [COLORS["muted"]] * 3

        fig.add_trace(go.Scatter(
            x=columns,
            y=y,
            mode="lines+markers",
            connectgaps=True,
            name="Vào top-k cuối" if kept else "Bị loại",
            legendgroup=group,
            showlegend=group not in legend_seen,
            line=dict(
                color=line_color,
                width=2.6 if kept else 1.2,
                dash="solid" if kept else "dot",
                shape="spline",
                smoothing=0.5,
            ),
            marker=dict(
                size=13 if kept else 9,
                color=marker_colors,
                line=dict(color=COLORS["bg"], width=2),
            ),
            opacity=1.0 if kept else 0.62,
            customdata=[[_short(row.get("title", ""), 60),
                         row.get("rrf_score"),
                         row.get("source", "")]] * 3,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "hạng: %{y}<br>"
                "RRF: %{customdata[1]:.5f}<br>"
                "<span style='opacity:.6'>%{customdata[2]}</span>"
                "<extra></extra>"
            ),
        ))
        legend_seen.add(group)

        if kept:
            fig.add_annotation(
                x=2, y=row["final_rank"],
                text=f"  {_short(row.get('title', ''), 34)}",
                showarrow=False, xanchor="left", align="left",
                font=dict(family=_FONT, size=10.5, color=COLORS["text_dim"]),
            )

    fig.update_xaxes(
        side="top", showgrid=False, zeroline=False,
        tickfont=dict(size=12.5),
        range=[-0.35, 2.9],  # room for the right-hand labels
    )
    fig.update_yaxes(
        autorange="reversed", dtick=1, range=[max_rank + 0.5, 0.5],
        title=dict(text="thứ hạng (1 = tốt nhất)",
                   font=dict(size=11, color=COLORS["faint"])),
        gridcolor=COLORS["border_soft"], zeroline=False,
        tickfont=dict(family=_MONO, size=11),
    )
    return _base_layout(fig, height=max(330, 150 + 34 * min(len(rows), 10)))


# --- 2. score comparison -----------------------------------------------------
def score_comparison_bars(trace: dict[str, Any]) -> go.Figure:
    """Grouped bars: dense cosine vs BM25 vs RRF score per candidate chunk.

    BM25 and RRF are min-max scaled to 0–1 because raw BM25 lives around 0–10
    and raw RRF around 0.015–0.033; plotted raw, two of the three series would be
    flat lines. The hover keeps the untouched values so nothing is hidden.
    """
    rows = list(trace.get("rank_movement") or [])
    if not rows:
        return _empty("Trace không có điểm số để so sánh.", height=320)

    rows = rows[:10]
    labels = [
        f"#{row['final_rank']}" if row.get("final_rank") else "—"
        for row in rows
    ]
    # Distinguish duplicate "—" labels so bars do not merge into one category.
    seen: dict[str, int] = {}
    for i, label in enumerate(labels):
        seen[label] = seen.get(label, 0) + 1
        if label == "—":
            labels[i] = "loại " + "·" * seen[label]

    titles = [_short(row.get("title", ""), 52) for row in rows]
    dense_raw = [row.get("dense_score") for row in rows]
    bm25_raw = [row.get("bm25_score") for row in rows]
    rrf_raw = [row.get("rrf_score") for row in rows]

    def _scaled(values: list[Any]) -> list[Any]:
        present = [v for v in values if v is not None]
        top = max(present) if present else 1.0
        top = top or 1.0
        return [None if v is None else v / top for v in values]

    series = [
        ("Dense (cosine thô)", dense_raw, dense_raw, COLORS["dense"], ".4f"),
        ("BM25 (chuẩn hoá)", _scaled(bm25_raw), bm25_raw, COLORS["bm25"], ".2f"),
        ("RRF (chuẩn hoá)", _scaled(rrf_raw), rrf_raw, COLORS["hybrid"], ".5f"),
    ]

    fig = go.Figure()
    for name, values, raw, color, fmt in series:
        fig.add_trace(go.Bar(
            x=labels, y=values, name=name,
            marker=dict(color=color, line=dict(width=0)),
            opacity=0.92,
            customdata=[[t, r if r is not None else float("nan")]
                        for t, r in zip(titles, raw)],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + name + ": %{customdata[1]:" + fmt + "}<extra></extra>"
            ),
        ))

    threshold = trace.get("score_threshold")
    if threshold is not None:
        # Only the dense series is on the same scale as the threshold, so the
        # line is annotated as such rather than presented as a global cut-off.
        fig.add_hline(
            y=float(threshold), line=dict(color=COLORS["fallback"], width=1.2,
                                          dash="dash"),
            annotation_text=f"ngưỡng dense {float(threshold):.2f}",
            annotation_position="top right",
            annotation_font=dict(size=10, color=COLORS["fallback"], family=_FONT),
        )

    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        title=dict(text="hạng cuối của chunk",
                   font=dict(size=11, color=COLORS["faint"])),
        tickfont=dict(family=_MONO, size=11),
    )
    fig.update_yaxes(
        gridcolor=COLORS["border_soft"], zeroline=False, rangemode="tozero",
        title=dict(text="điểm (0–1)", font=dict(size=11, color=COLORS["faint"])),
        tickfont=dict(family=_MONO, size=11),
    )
    return _base_layout(fig, height=330)


# --- 3. threshold gauge ------------------------------------------------------
def threshold_gauge(trace: dict[str, Any]) -> go.Figure:
    """Indicator gauge: best dense cosine against the fallback threshold.

    The red arc is the fallback zone and the green arc is the hybrid zone, so
    the single needle position answers "why did the pipeline choose this branch"
    without reading a number.
    """
    best = float(trace.get("best_dense_score") or 0.0)
    threshold = float(trace.get("score_threshold") or 0.0)
    passed = best >= threshold
    accent = COLORS["hybrid"] if passed else COLORS["fallback"]

    # This corpus's embedding model (multilingual-e5-small) squeezes every
    # cosine into roughly 0.82-0.92, so a fixed 0-1 arc would render the
    # pass/fail margin as a couple of degrees. The floor drops to whatever the
    # data needs and the tick labels state it, so the axis is compressed but
    # never mislabelled.
    low = min(best, threshold)
    axis_min = 0.0 if low < 0.55 else round(max(0.0, low - 0.12), 2)
    span = 1.0 - axis_min

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=best,
        number=dict(font=dict(family=_MONO, size=34, color=accent),
                    valueformat=".4f"),
        delta=dict(
            reference=threshold, valueformat=".4f",
            increasing=dict(color=COLORS["ok"]),
            decreasing=dict(color=COLORS["fallback"]),
            font=dict(family=_MONO, size=13),
        ),
        title=dict(
            text="<span style='font-size:12px'>điểm dense cao nhất "
                 f"<span style='color:{COLORS['faint']}'>vs ngưỡng "
                 f"{threshold:.3f} · thang {axis_min:.2f}–1.00</span></span>",
            font=dict(family=_FONT, color=COLORS["muted"]),
        ),
        gauge=dict(
            axis=dict(range=[axis_min, 1], tickwidth=1,
                      tickcolor=COLORS["border"],
                      tickfont=dict(family=_MONO, size=10,
                                    color=COLORS["faint"]),
                      dtick=round(span / 5, 3)),
            bar=dict(color=accent, thickness=0.26),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[axis_min, threshold], color="rgba(251,113,133,.20)"),
                dict(range=[threshold, 1], color="rgba(52,211,153,.16)"),
            ],
            threshold=dict(
                line=dict(color=COLORS["text"], width=2.5),
                thickness=0.9, value=threshold,
            ),
        ),
    ))
    return _base_layout(fig, height=250)


# --- 4. latency --------------------------------------------------------------
def latency_breakdown(trace: dict[str, Any]) -> go.Figure:
    """Horizontal stacked bar of where the wall-clock time went.

    Kept as one row on a linear axis on purpose: the point the audience should
    leave with is that generation dominates and that BM25 plus RRF are free.
    Splitting it into per-stage rows would hide exactly that.
    """
    stages = {s.get("name"): s for s in (trace.get("stages") or [])}
    segments = [
        ("Dense", float(stages.get("dense", {}).get("elapsed_ms") or 0.0),
         COLORS["dense"]),
        ("BM25", float(stages.get("bm25", {}).get("elapsed_ms") or 0.0),
         COLORS["bm25"]),
        ("RRF", float(stages.get("rrf", {}).get("elapsed_ms") or 0.0),
         COLORS["hybrid"]),
        ("Fallback", float(stages.get("fallback", {}).get("elapsed_ms") or 0.0),
         COLORS["fallback"]),
        ("LLM", float(trace.get("llm_elapsed_ms") or 0.0), COLORS["ok"]),
    ]
    total = sum(value for _, value, _ in segments)
    if total <= 0:
        return _empty("Chưa có số liệu thời gian.", height=170)

    fig = go.Figure()
    for name, value, color in segments:
        share = value / total
        fig.add_trace(go.Bar(
            x=[value], y=["thời gian"], orientation="h", name=name,
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{name} {value:.0f}ms"] if share > 0.08 else [""],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(family=_MONO, size=11, color=COLORS["bg"]),
            hovertemplate=f"{name}: %{{x:.1f}} ms ({share:.1%})<extra></extra>",
        ))

    retrieval = total - segments[-1][1]
    fig.add_annotation(
        x=1, y=1.7, xref="paper", yref="y", xanchor="right", showarrow=False,
        text=(f"tổng <b>{total:.0f} ms</b>  ·  truy hồi {retrieval:.0f} ms "
              f"({retrieval / total:.1%})"),
        font=dict(family=_FONT, size=11, color=COLORS["muted"]),
    )

    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(
        showgrid=True, gridcolor=COLORS["border_soft"], zeroline=False,
        title=dict(text="mili-giây", font=dict(size=11, color=COLORS["faint"])),
        tickfont=dict(family=_MONO, size=10.5),
    )
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return _base_layout(fig, height=185)
