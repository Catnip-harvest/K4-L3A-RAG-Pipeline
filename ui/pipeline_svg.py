"""Hand-written inline SVG of the retrieval pipeline, drawn per query.

WHY hand-written SVG instead of a diagram library: the diagram has to carry the
*actual* numbers from this query's trace — candidate counts, per-branch latency,
the RRF k, and the exact comparison that opened the threshold gate — and it has
to dim the branch that was not taken. No generic graph library gives that
without fighting it, and a hand-written string has no runtime dependency at all.

Animation is real SMIL (``<animate>``), not JavaScript, because Streamlit's
``st.html`` strips scripts. If a sanitiser ever removes the SMIL elements the
diagram degrades to a correct static drawing — every element already carries the
final value of whatever it animates.

Legibility in light *and* dark is handled by painting the panel background
inside the SVG: the drawing never inherits the page colour, so it reads the same
whatever theme surrounds it. Call with ``dark=False`` for a light deck.
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

# Canvas geometry. One place to change if the diagram is ever re-laid-out.
_W, _H = 1180, 470

_DARK = {
    "panel": "#0C1424",
    "panel_edge": "#1F2C46",
    "box": "#131F35",
    "box_edge": "#26344F",
    "text": "#E9F0FC",
    "muted": "#93A6C6",
    "faint": "#5F719055",
    "idle": "#3A4A68",
    "dense": "#7C5CFF",
    "bm25": "#22D3EE",
    "hybrid": "#FFB020",
    "fallback": "#FB7185",
    "ok": "#34D399",
    "grid": "#18243B",
}

_LIGHT = {
    "panel": "#FFFFFF",
    "panel_edge": "#D9E0EC",
    "box": "#F5F7FC",
    "box_edge": "#D3DBE9",
    "text": "#0C1626",
    "muted": "#55647F",
    "faint": "#8899B355",
    "idle": "#BFCADB",
    "dense": "#6035F2",
    "bm25": "#0B93AE",
    "hybrid": "#B97400",
    "fallback": "#D8385C",
    "ok": "#0F855C",
    "grid": "#EEF2F8",
}

_SANS = "Be Vietnam Pro, Inter, Segoe UI, sans-serif"
_MONO = "JetBrains Mono, SFMono-Regular, Consolas, monospace"


def _esc(value: Any) -> str:
    """XML-escape any value for safe interpolation into the SVG."""
    return escape(str(value))


def _ms(value: Any) -> str:
    """Format an elapsed_ms field, tolerating None."""
    try:
        return f"{float(value):.0f} ms"
    except (TypeError, ValueError):
        return "— ms"


def _stage(trace: dict[str, Any], name: str) -> dict[str, Any]:
    """Look a stage up by name; missing stages degrade to an empty dict."""
    for stage in trace.get("stages") or []:
        if stage.get("name") == name:
            return stage
    return {}


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12,
    fill: str = "#fff",
    weight: int = 400,
    anchor: str = "start",
    mono: bool = False,
    opacity: float = 1.0,
    spacing: float | None = None,
) -> str:
    family = _MONO if mono else _SANS
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'opacity="{opacity}"{extra}>{_esc(content)}</text>'
    )


def _node(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    subtitle: str,
    metric: str,
    accent: str,
    palette: dict[str, str],
    active: bool,
    badge: str = "",
) -> str:
    """One pipeline box: title, subtitle, a monospaced metric and a colour rail."""
    opacity = 1.0 if active else 0.38
    stroke = accent if active else palette["idle"]
    width = 1.6 if active else 1.0
    glow = (
        f'<rect x="{x - 3}" y="{y - 3}" width="{w + 6}" height="{h + 6}" rx="17" '
        f'fill="none" stroke="{accent}" stroke-width="1" opacity="0.16"/>'
        if active
        else ""
    )
    return (
        f'<g opacity="{opacity}">{glow}'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
        f'fill="{palette["box"]}" stroke="{stroke}" stroke-width="{width}"/>'
        f'<rect x="{x}" y="{y + 12}" width="3" height="{h - 24}" rx="2" fill="{accent}"/>'
        + _text(x + 16, y + 26, title, size=13.5, fill=palette["text"], weight=600)
        + _text(x + 16, y + 45, subtitle, size=10.5, fill=palette["muted"])
        + _text(x + 16, y + h - 14, metric, size=12, fill=accent, weight=600, mono=True)
        + (
            _text(x + w - 14, y + 26, badge, size=9.5, fill=accent, weight=700,
                  anchor="end", spacing=0.8)
            if badge
            else ""
        )
        + "</g>"
    )


def _edge(
    path: str,
    *,
    color: str,
    active: bool,
    palette: dict[str, str],
    label: str = "",
    label_xy: tuple[float, float] | None = None,
    label_anchor: str = "middle",
    dur: str = "1.2s",
) -> str:
    """A pipeline edge. Active edges are thick and carry a flowing dash."""
    if active:
        base = (
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" '
            f'opacity="0.34" stroke-linecap="round"/>'
        )
        # The dash offset animation is what makes the taken path read as "flow".
        flow = (
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-dasharray="9 15" stroke-dashoffset="0">'
            f'<animate attributeName="stroke-dashoffset" from="24" to="0" '
            f'dur="{dur}" repeatCount="indefinite"/></path>'
        )
        marker = ""
    else:
        base = (
            f'<path d="{path}" fill="none" stroke="{palette["idle"]}" '
            f'stroke-width="1.3" stroke-dasharray="4 6" opacity="0.75"/>'
        )
        flow = marker = ""

    text = ""
    if label and label_xy:
        lx, ly = label_xy
        text = _text(
            lx, ly, label,
            size=10.5,
            fill=color if active else palette["muted"],
            weight=600 if active else 400,
            anchor=label_anchor,
            mono=True,
            opacity=1.0 if active else 0.6,
        )
    return base + flow + marker + text


def _arrow(x: float, y: float, color: str, active: bool, palette: dict[str, str]) -> str:
    """Small triangular arrow head at an edge's terminus."""
    fill = color if active else palette["idle"]
    opacity = 1.0 if active else 0.6
    return (
        f'<path d="M {x} {y} l -7 -4.5 l 0 9 z" fill="{fill}" opacity="{opacity}"/>'
    )


def _gate(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    best: float,
    threshold: float,
    passed: bool,
    verdict: str,
    palette: dict[str, str],
) -> str:
    """The threshold gate — the single most important element in the diagram.

    It shows the literal comparison that decided the route, plus a small track
    where the score dot sits against the threshold tick, so the audience can see
    *how close* the call was rather than just its outcome.
    """
    accent = palette["hybrid"] if passed else palette["fallback"]
    sign = "≥" if passed else "<"

    track_x, track_w = x + 18, w - 36
    track_y = y + h - 26
    pos = max(0.0, min(1.0, best))
    thr = max(0.0, min(1.0, threshold))

    return (
        f'<g>'
        # halo ring; opacity animation is decorative and degrades to a static ring
        f'<rect x="{x - 5}" y="{y - 5}" width="{w + 10}" height="{h + 10}" rx="20" '
        f'fill="none" stroke="{accent}" stroke-width="1.2" opacity="0.28">'
        f'<animate attributeName="opacity" values="0.12;0.42;0.12" dur="2.6s" '
        f'repeatCount="indefinite"/></rect>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
        f'fill="{palette["box"]}" stroke="{accent}" stroke-width="1.8"/>'
        + _text(x + 18, y + 24, "CỔNG NGƯỠNG", size=9.5, fill=palette["muted"],
                weight=700, spacing=1.4)
        + _text(x + w - 18, y + 24, "vs ngưỡng", size=9, fill=palette["muted"],
                anchor="end")
        + _text(x + 18, y + 54, f"{best:.3f} {sign} {threshold:.3f}", size=22,
                fill=accent, weight=700, mono=True)
        + _text(x + 18, y + 76, f"→ {verdict}", size=12.5, fill=palette["text"],
                weight=600, mono=True)
        # 0..1 track with the threshold tick and the actual score dot
        + f'<rect x="{track_x}" y="{track_y}" width="{track_w}" height="5" rx="3" '
          f'fill="{palette["fallback"]}" opacity="0.22"/>'
        + f'<rect x="{track_x + track_w * thr}" y="{track_y}" '
          f'width="{track_w * (1 - thr)}" height="5" rx="3" '
          f'fill="{palette["ok"]}" opacity="0.28"/>'
        + f'<rect x="{track_x + track_w * thr - 1}" y="{track_y - 5}" width="2" '
          f'height="15" rx="1" fill="{palette["muted"]}"/>'
        + f'<circle cx="{track_x + track_w * pos}" cy="{track_y + 2.5}" r="5.5" '
          f'fill="{accent}" stroke="{palette["panel"]}" stroke-width="2">'
          f'<animate attributeName="r" values="5;6.5;5" dur="2.6s" '
          f'repeatCount="indefinite"/></circle>'
        + _text(track_x, track_y + 22, "0.0", size=8.5, fill=palette["muted"], mono=True)
        + _text(track_x + track_w, track_y + 22, "1.0", size=8.5,
                fill=palette["muted"], mono=True, anchor="end")
        + "</g>"
    )


def _legend(y: float, palette: dict[str, str]) -> str:
    """Colour key. Five seconds of reading buys the whole rest of the UI."""
    items = [
        (palette["dense"], "Dense — vector"),
        (palette["bm25"], "BM25 — từ khoá"),
        (palette["hybrid"], "Hybrid / RRF"),
        (palette["fallback"], "Fallback / từ chối"),
    ]
    out = [
        _text(24, y + 4, "MÃ MÀU", size=9, fill=palette["muted"], weight=700, spacing=1.4)
    ]
    x = 96.0
    for color, label in items:
        out.append(
            f'<rect x="{x}" y="{y - 4}" width="16" height="4" rx="2" fill="{color}"/>'
        )
        out.append(_text(x + 23, y + 4, label, size=10.5, fill=palette["muted"]))
        x += 30 + len(label) * 6.6
    out.append(
        _text(_W - 24, y + 4, "nét đứt mờ = nhánh không được chọn", size=10,
              fill=palette["muted"], anchor="end", opacity=0.8)
    )
    return "".join(out)


def render_pipeline_svg(trace: dict[str, Any], *, dark: bool = True) -> str:
    """Return the full inline SVG for one query's trace.

    Every number on the diagram comes from ``trace``; nothing is hard-coded.
    """
    palette = _DARK if dark else _LIGHT

    dense = _stage(trace, "dense")
    bm25 = _stage(trace, "bm25")
    rrf = _stage(trace, "rrf")
    fallback = _stage(trace, "fallback")

    use_reranking = bool(trace.get("use_reranking", True))
    best = float(trace.get("best_dense_score") or 0.0)
    threshold = float(trace.get("score_threshold") or 0.0)
    decision = str(trace.get("decision") or "hybrid")
    fell_back = decision == "pageindex" or bool(fallback.get("triggered"))
    passed = not fell_back
    refused = bool(trace.get("refused")) or decision == "none"

    gen_active = passed and not refused
    fb_active = fell_back

    c_dense = palette["dense"]
    c_bm25 = palette["bm25"]
    c_hybrid = palette["hybrid"]
    c_fallback = palette["fallback"]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'width="100%" role="img" aria-label="Sơ đồ pipeline truy hồi" '
        f'style="max-width:100%;height:auto;display:block">',
        f'<rect x="0" y="0" width="{_W}" height="{_H}" rx="18" '
        f'fill="{palette["panel"]}" stroke="{palette["panel_edge"]}"/>',
    ]

    # --- header -------------------------------------------------------------
    parts.append(
        _text(24, 32, "ĐƯỜNG ĐI CỦA TRUY VẤN", size=11, fill=palette["muted"],
              weight=700, spacing=1.8)
    )
    head_color = c_fallback if fell_back else c_hybrid
    head_label = (
        "TỪ CHỐI AN TOÀN" if refused
        else ("NHÁNH FALLBACK" if fell_back else "NHÁNH HYBRID")
    )
    parts.append(
        f'<rect x="{_W - 24 - 168}" y="16" width="168" height="24" rx="12" '
        f'fill="{head_color}" opacity="0.14"/>'
    )
    parts.append(
        f'<circle cx="{_W - 24 - 152}" cy="28" r="4" fill="{head_color}">'
        f'<animate attributeName="opacity" values="1;0.25;1" dur="1.8s" '
        f'repeatCount="indefinite"/></circle>'
    )
    parts.append(
        _text(_W - 24 - 12, 32, head_label, size=10, fill=head_color, weight=700,
              anchor="end", spacing=1.0)
    )
    parts.append(
        _text(24, 52, f"tổng {_ms(trace.get('elapsed_ms'))} truy hồi"
              + (f" + {_ms(trace.get('llm_elapsed_ms'))} sinh câu trả lời"
                 if trace.get("llm_elapsed_ms") else ""),
              size=10.5, fill=palette["muted"], mono=True)
    )

    # --- nodes --------------------------------------------------------------
    qx, qy, qw, qh = 24, 188, 140, 84
    dx, dy, dw, dh = 214, 96, 214, 92
    bx, by, bw, bh = 214, 282, 214, 92
    rx, ry, rw, rh = 492, 184, 168, 92
    gx, gy, gw, gh = 722, 170, 196, 120
    ox, oy, ow, oh = 964, 96, 192, 92
    fx, fy, fw, fh = 964, 282, 192, 92

    query_text = str(trace.get("query") or "")
    shown = query_text if len(query_text) <= 22 else query_text[:21] + "…"
    parts.append(
        _node(qx, qy, qw, qh, title="Câu hỏi", subtitle=shown, metric=
              f"top_k = {trace.get('top_k', '—')}", accent=palette["text"],
              palette=palette, active=True)
    )
    parts.append(
        _node(dx, dy, dw, dh, title="Dense (vector)",
              subtitle="cosine trên ChromaDB",
              metric=f"{dense.get('count', 0)} ứng viên · {_ms(dense.get('elapsed_ms'))}",
              accent=c_dense, palette=palette, active=True, badge="BƯỚC 1A")
    )
    parts.append(
        _node(bx, by, bw, bh, title="BM25 (từ khoá)",
              subtitle="rank_bm25 trên toàn corpus",
              metric=f"{bm25.get('count', 0)} ứng viên · {_ms(bm25.get('elapsed_ms'))}",
              accent=c_bm25, palette=palette, active=use_reranking, badge="BƯỚC 1B")
    )
    rrf_k = rrf.get("k", 60)
    parts.append(
        _node(rx, ry, rw, rh,
              title="RRF fusion" if use_reranking else "RRF (đã tắt)",
              subtitle=f"k = {rrf_k}" if use_reranking else "chỉ dùng dense",
              metric=(f"{rrf.get('count', 0)} gộp · {_ms(rrf.get('elapsed_ms'))}"
                      if use_reranking else "bỏ qua"),
              accent=c_hybrid, palette=palette, active=use_reranking, badge="BƯỚC 2")
    )
    parts.append(
        _gate(gx, gy, gw, gh, best=best, threshold=threshold, passed=passed,
              verdict=("hybrid" if use_reranking else "dense") if passed
              else "pageindex",
              palette=palette)
    )
    parts.append(
        _node(ox, oy, ow, oh, title="Sinh câu trả lời",
              subtitle=f"{trace.get('provider', 'llm')} · {trace.get('model', '')}"[:34],
              metric=(f"{trace.get('context_chars', 0)} ký tự context"
                      if gen_active else "không chạy"),
              accent=palette["ok"] if gen_active else palette["idle"],
              palette=palette, active=gen_active, badge="BƯỚC 4")
    )
    fb_error = fallback.get("error")
    parts.append(
        _node(fx, fy, fw, fh, title="Fallback",
              subtitle="vectorless · " + str(fallback.get("provider") or "pageindex"),
              metric=("lỗi: " + str(fb_error)[:18] if fb_active and fb_error
                      else (f"{fallback.get('count', 0)} kết quả · "
                            f"{_ms(fallback.get('elapsed_ms'))}" if fb_active
                            else "không kích hoạt")),
              accent=c_fallback, palette=palette, active=fb_active, badge="BƯỚC 3B")
    )

    # --- edges --------------------------------------------------------------
    # query -> dense
    parts.append(_edge(
        f"M {qx + qw} 214 C {qx + qw + 30} 214, {dx - 30} 142, {dx - 8} 142",
        color=c_dense, active=True, palette=palette,
        label=f"{dense.get('count', 0)} kq", label_xy=(dx - 26, 132)))
    parts.append(_arrow(dx - 1, 142, c_dense, True, palette))
    # query -> bm25
    parts.append(_edge(
        f"M {qx + qw} 246 C {qx + qw + 30} 246, {bx - 30} 328, {bx - 8} 328",
        color=c_bm25, active=use_reranking, palette=palette,
        label=f"{bm25.get('count', 0)} kq", label_xy=(bx - 26, 348)))
    parts.append(_arrow(bx - 1, 328, c_bm25, use_reranking, palette))
    # dense -> rrf
    parts.append(_edge(
        f"M {dx + dw} 142 C {dx + dw + 34} 142, {rx - 34} 208, {rx - 8} 208",
        color=c_dense, active=use_reranking, palette=palette))
    parts.append(_arrow(rx - 1, 208, c_dense, use_reranking, palette))
    # bm25 -> rrf
    parts.append(_edge(
        f"M {bx + bw} 328 C {bx + bw + 34} 328, {rx - 34} 252, {rx - 8} 252",
        color=c_bm25, active=use_reranking, palette=palette))
    parts.append(_arrow(rx - 1, 252, c_bm25, use_reranking, palette))
    # rrf -> gate
    parts.append(_edge(
        f"M {rx + rw} 230 L {gx - 8} 230",
        color=c_hybrid, active=use_reranking, palette=palette,
        label="top-k" if use_reranking else "", label_xy=(rx + rw + 31, 216)))
    parts.append(_arrow(gx - 1, 230, c_hybrid, use_reranking, palette))
    # dense bypass (only when reranking is off)
    if not use_reranking:
        parts.append(_edge(
            f"M {dx + dw} 120 C {dx + dw + 120} 70, {gx - 120} 96, {gx + 40} {gy - 8}",
            color=c_dense, active=True, palette=palette,
            label="bỏ qua RRF — chỉ dense", label_xy=(rx + 60, 72)))
        parts.append(
            f'<path d="M {gx + 40} {gy - 1} l -4.5 -7 l 9 0 z" fill="{c_dense}"/>')
    # gate -> generation
    parts.append(_edge(
        f"M {gx + gw} 206 C {gx + gw + 26} 206, {ox - 26} 142, {ox - 8} 142",
        color=palette["ok"] if gen_active else palette["idle"], active=gen_active,
        palette=palette,
        label="đạt ngưỡng" if gen_active else "", label_xy=(ox - 14, 128),
        label_anchor="end"))
    parts.append(_arrow(ox - 1, 142, palette["ok"], gen_active, palette))
    # gate -> fallback
    parts.append(_edge(
        f"M {gx + gw} 254 C {gx + gw + 26} 254, {fx - 26} 328, {fx - 8} 328",
        color=c_fallback, active=fb_active, palette=palette,
        label="dưới ngưỡng" if fb_active else "", label_xy=(fx - 14, 352),
        label_anchor="end"))
    parts.append(_arrow(fx - 1, 328, c_fallback, fb_active, palette))

    # --- decision caption ---------------------------------------------------
    reason = str(trace.get("decision_reason") or "")
    if len(reason) > 118:
        reason = reason[:117] + "…"
    parts.append(
        f'<rect x="24" y="396" width="{_W - 48}" height="30" rx="9" '
        f'fill="{head_color}" opacity="0.08"/>'
    )
    parts.append(
        f'<rect x="24" y="396" width="3" height="30" rx="2" fill="{head_color}"/>'
    )
    parts.append(_text(40, 415, reason, size=11.5, fill=palette["text"], weight=500))

    parts.append(_legend(450, palette))
    parts.append("</svg>")
    return "".join(parts)
