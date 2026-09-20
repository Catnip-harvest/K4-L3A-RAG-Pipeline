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

The drawing paints its own panel background, so it never inherits the page
colour and reads the same whatever surrounds it. There is exactly one palette
and it is light: the deck is projected in a lit classroom, where a dark panel
loses every hairline.

COLOUR RULE, which the rest of this module obeys without exception: a hue
appears only where it says which retrieval path something belongs to — indigo
for dense, teal for BM25, amber for RRF fusion, rose for the vectorless
fallback, green for a grounded answer. Everything structural — panels, boxes,
ticks, arrows that carry no signal, labels — is a neutral grey. A branch that
was not taken loses its hue and becomes grey too; that *is* the dimming, and it
replaces the old opacity fade, which on a light panel dropped text below the
legibility floor.
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

# Canvas geometry. One place to change if the diagram is ever re-laid-out.
_W, _H = 1180, 470

# The palette comes straight from ui/theme.py. This module used to carry its
# own copy "kept in sync by hand", which is a promise that survives exactly one
# palette change; the diagram, the Plotly figures and the CSS now all read the
# same dict, so a colour can only be changed in one place.
from .theme import COLORS as _C  # noqa: E402

# Light tint behind a status pill or caption band, keyed by its signal colour.
_SOFT: dict[str, str] = {
    _C["dense"]: _C["dense_soft"],
    _C["bm25"]: _C["bm25_soft"],
    _C["hybrid"]: _C["hybrid_soft"],
    _C["fallback"]: _C["fallback_soft"],
}

# Smallest type on the canvas. Below this a label stops surviving projection,
# so every _text call is at or above it.
_MIN_SIZE = 11.0

_SHADOW = "url(#pipeShadow)"

# Chrome for the branch that was not taken. `muted` held at 0.8 over the panel
# measures 3.25:1 — it clears the 3:1 floor for a line that carries meaning,
# and still sits well under the 5-6:1 of a live, coloured edge, so the eye
# still goes to the path the query actually took.
_IDLE = _C["muted"]
_IDLE_ALPHA = 0.8

_SANS = "Be Vietnam Pro, Inter, Segoe UI, sans-serif"
_MONO = "JetBrains Mono, SFMono-Regular, Consolas, monospace"


def _esc(value: Any) -> str:
    """XML-escape any value for safe interpolation into the SVG."""
    return escape(str(value))


def _soft(color: str) -> str:
    """The light tint that belongs to a signal colour."""
    return _SOFT.get(color, _C["surface_3"])


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
    fill: str = _C["text"],
    weight: int = 400,
    anchor: str = "start",
    mono: bool = False,
    opacity: float = 1.0,
    spacing: float | None = None,
) -> str:
    family = _MONO if mono else _SANS
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    # A caller asking for 9px would be unreadable from the back of the room.
    size = max(size, _MIN_SIZE)
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'opacity="{opacity}"{extra}>{_esc(content)}</text>'
    )


def _defs() -> str:
    """The one filter the drawing uses: a soft slate shadow, not a glow.

    A glow needs a dark backdrop to register. On a white panel the same effect
    reads as dirt, so active boxes are lifted with a low-opacity slate shadow
    instead.
    """
    return (
        '<defs><filter id="pipeShadow" x="-10%" y="-25%" width="120%" height="160%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#101828" '
        'flood-opacity="0.08"/></filter></defs>'
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
    """One pipeline box: title, subtitle, a monospaced metric and a colour rail.

    An inactive box keeps full opacity and drops to greys instead. Fading it to
    38% — which is what the dark build did — put every label under 3:1 here.
    """
    if active:
        fill = palette["surface"]
        edge = accent
        rail = accent
        title_fill = palette["text"]
        sub_fill = palette["muted"]
        metric_fill = accent
        badge_fill = accent
        width = 1.6
        extra = f' filter="{_SHADOW}"'
    else:
        fill = palette["surface_3"]
        edge = _IDLE
        rail = palette["faint"]
        # surface_3 is light enough that `muted` lands at 4.4:1, just under AA,
        # so everything on an inactive box uses text_dim.
        title_fill = sub_fill = metric_fill = badge_fill = palette["text_dim"]
        width = 1.2
        extra = f' stroke-opacity="{_IDLE_ALPHA}"'

    return (
        "<g>"
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
        f'fill="{fill}" stroke="{edge}" stroke-width="{width}"{extra}/>'
        f'<rect x="{x}" y="{y + 12}" width="3.5" height="{h - 24}" rx="2" fill="{rail}"/>'
        + _text(x + 18, y + 27, title, size=14, fill=title_fill, weight=600)
        + _text(x + 18, y + 47, subtitle, size=11, fill=sub_fill)
        + _text(x + 18, y + h - 14, metric, size=12, fill=metric_fill,
                weight=600, mono=True)
        + (
            _text(x + w - 14, y + 27, badge, size=11, fill=badge_fill, weight=700,
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
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6" '
            f'opacity="0.3" stroke-linecap="round"/>'
        )
        # The dash offset animation is what makes the taken path read as "flow":
        # a full-strength dash travelling over its own light tint.
        flow = (
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6" '
            f'stroke-linecap="round" stroke-dasharray="9 15" stroke-dashoffset="0">'
            f'<animate attributeName="stroke-dashoffset" from="24" to="0" '
            f'dur="{dur}" repeatCount="indefinite"/></path>'
        )
    else:
        base = (
            f'<path d="{path}" fill="none" stroke="{_IDLE}" '
            f'stroke-opacity="{_IDLE_ALPHA}" stroke-width="1.6" '
            f'stroke-dasharray="4 6"/>'
        )
        flow = ""

    text = ""
    if label and label_xy:
        lx, ly = label_xy
        text = _text(
            lx, ly, label,
            size=11,
            fill=color if active else palette["muted"],
            weight=600 if active else 400,
            anchor=label_anchor,
            mono=True,
        )
    return base + flow + text


def _arrow(x: float, y: float, color: str, active: bool) -> str:
    """Small triangular arrow head at an edge's terminus."""
    fill = color if active else _IDLE
    alpha = "" if active else f' fill-opacity="{_IDLE_ALPHA}"'
    return f'<path d="M {x} {y} l -7 -4.5 l 0 9 z" fill="{fill}"{alpha}/>'


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
        # Halo: a light tint of the outcome colour that breathes by fading in.
        # On a dark panel this was a brightening ring; on white a ring that
        # brightens disappears, so the tint deepens instead.
        f'<rect x="{x - 6}" y="{y - 6}" width="{w + 12}" height="{h + 12}" rx="22" '
        f'fill="{_soft(accent)}" stroke="{accent}" stroke-width="1.2" '
        f'stroke-opacity="0.35" opacity="1">'
        f'<animate attributeName="opacity" values="0.45;1;0.45" dur="2.6s" '
        f'repeatCount="indefinite"/></rect>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
        f'fill="{palette["surface"]}" stroke="{accent}" stroke-width="1.8" '
        f'filter="{_SHADOW}"/>'
        + _text(x + 18, y + 26, "CỔNG NGƯỠNG", size=11, fill=palette["text_dim"],
                weight=700, spacing=1.4)
        + _text(x + w - 18, y + 26, "vs ngưỡng", size=11, fill=palette["muted"],
                anchor="end")
        + _text(x + 18, y + 56, f"{best:.3f} {sign} {threshold:.3f}", size=20,
                fill=accent, weight=700, mono=True)
        + _text(x + 18, y + 78, f"→ {verdict}", size=12.5, fill=palette["text"],
                weight=600, mono=True)
        # 0..1 track with the threshold tick and the actual score dot
        + f'<rect x="{track_x}" y="{track_y}" width="{track_w}" height="5" rx="3" '
          f'fill="{palette["fallback_soft"]}"/>'
        # No ok_soft token exists, so the pass zone is `ok` held at a tint.
        + f'<rect x="{track_x + track_w * thr}" y="{track_y}" '
          f'width="{track_w * (1 - thr)}" height="5" rx="3" '
          f'fill="{palette["ok"]}" opacity="0.18"/>'
        + f'<rect x="{track_x + track_w * thr - 1.25}" y="{track_y - 5}" width="2.5" '
          f'height="15" rx="1" fill="{palette["text_dim"]}"/>'
        + f'<circle cx="{track_x + track_w * pos}" cy="{track_y + 2.5}" r="5.5" '
          f'fill="{accent}" stroke="{palette["surface"]}" stroke-width="2">'
          f'<animate attributeName="r" values="5;6.5;5" dur="2.6s" '
          f'repeatCount="indefinite"/></circle>'
        + _text(track_x, track_y + 22, "0.0", size=11, fill=palette["muted"], mono=True)
        + _text(track_x + track_w, track_y + 22, "1.0", size=11,
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
        _text(24, y + 4, "MÃ MÀU", size=11, fill=palette["text_dim"], weight=700,
              spacing=1.4)
    ]
    x = 104.0
    for color, label in items:
        out.append(
            f'<rect x="{x}" y="{y - 4}" width="18" height="5" rx="2.5" fill="{color}"/>'
        )
        out.append(_text(x + 25, y + 4, label, size=11, fill=palette["muted"]))
        x += 34 + len(label) * 6.9
    out.append(
        _text(_W - 24, y + 4, "nét đứt xám = nhánh không được chọn", size=11,
              fill=palette["muted"], anchor="end")
    )
    return "".join(out)


def render_pipeline_svg(trace: dict[str, Any], *, dark: bool = True) -> str:
    """Return the full inline SVG for one query's trace.

    Every number on the diagram comes from ``trace``; nothing is hard-coded.

    ``dark`` is accepted and ignored. The deck has one palette now, and the
    argument stays only so existing call sites keep working.
    """
    palette = _C

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
        _defs(),
        f'<rect x="0" y="0" width="{_W}" height="{_H}" rx="18" '
        f'fill="{palette["surface_2"]}" stroke="{palette["border"]}"/>',
    ]

    # --- header -------------------------------------------------------------
    parts.append(
        _text(24, 32, "ĐƯỜNG ĐI CỦA TRUY VẤN", size=11, fill=palette["text_dim"],
              weight=700, spacing=1.8)
    )
    # Rose covers both ways the hybrid path can fail to answer: the vectorless
    # fallback and a safe refusal. The old build painted a refusal amber, which
    # said "hybrid path, all good" next to the words "TỪ CHỐI AN TOÀN".
    head_color = c_fallback if (fell_back or refused) else c_hybrid
    head_label = (
        "TỪ CHỐI AN TOÀN" if refused
        else ("NHÁNH FALLBACK" if fell_back else "NHÁNH HYBRID")
    )
    parts.append(
        f'<rect x="{_W - 24 - 176}" y="14" width="176" height="26" rx="13" '
        f'fill="{_soft(head_color)}" stroke="{head_color}" stroke-opacity="0.3"/>'
    )
    parts.append(
        f'<circle cx="{_W - 24 - 158}" cy="27" r="4" fill="{head_color}" opacity="1">'
        f'<animate attributeName="opacity" values="0.4;1;0.4" dur="1.8s" '
        f'repeatCount="indefinite"/></circle>'
    )
    parts.append(
        _text(_W - 24 - 12, 31, head_label, size=11, fill=head_color, weight=700,
              anchor="end", spacing=1.0)
    )
    parts.append(
        _text(24, 54, f"tổng {_ms(trace.get('elapsed_ms'))} truy hồi"
              + (f" + {_ms(trace.get('llm_elapsed_ms'))} sinh câu trả lời"
                 if trace.get("llm_elapsed_ms") else ""),
              size=11, fill=palette["muted"], mono=True)
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
    # 11px in a 140-wide box: 16 characters is what actually fits.
    shown = query_text if len(query_text) <= 16 else query_text[:15] + "…"
    parts.append(
        _node(qx, qy, qw, qh, title="Câu hỏi", subtitle=shown, metric=
              f"top_k = {trace.get('top_k', '—')}", accent=palette["accent"],
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
              subtitle=f"{trace.get('provider', 'llm')} · {trace.get('model', '')}"[:26],
              metric=(f"{trace.get('context_chars', 0)} ký tự context"
                      if gen_active else "không chạy"),
              accent=palette["ok"], palette=palette, active=gen_active,
              badge="BƯỚC 4")
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
        label=f"{dense.get('count', 0)} kq", label_xy=(dx - 26, 130)))
    parts.append(_arrow(dx - 1, 142, c_dense, True))
    # query -> bm25
    parts.append(_edge(
        f"M {qx + qw} 246 C {qx + qw + 30} 246, {bx - 30} 328, {bx - 8} 328",
        color=c_bm25, active=use_reranking, palette=palette,
        label=f"{bm25.get('count', 0)} kq", label_xy=(bx - 26, 350)))
    parts.append(_arrow(bx - 1, 328, c_bm25, use_reranking))
    # dense -> rrf
    parts.append(_edge(
        f"M {dx + dw} 142 C {dx + dw + 34} 142, {rx - 34} 208, {rx - 8} 208",
        color=c_dense, active=use_reranking, palette=palette))
    parts.append(_arrow(rx - 1, 208, c_dense, use_reranking))
    # bm25 -> rrf
    parts.append(_edge(
        f"M {bx + bw} 328 C {bx + bw + 34} 328, {rx - 34} 252, {rx - 8} 252",
        color=c_bm25, active=use_reranking, palette=palette))
    parts.append(_arrow(rx - 1, 252, c_bm25, use_reranking))
    # rrf -> gate
    parts.append(_edge(
        f"M {rx + rw} 230 L {gx - 8} 230",
        color=c_hybrid, active=use_reranking, palette=palette,
        label="top-k" if use_reranking else "", label_xy=(rx + rw + 31, 214)))
    parts.append(_arrow(gx - 1, 230, c_hybrid, use_reranking))
    # dense bypass (only when reranking is off)
    if not use_reranking:
        parts.append(_edge(
            f"M {dx + dw} 120 C {dx + dw + 120} 70, {gx - 120} 96, {gx + 40} {gy - 8}",
            color=c_dense, active=True, palette=palette,
            label="bỏ qua RRF — chỉ dense", label_xy=(rx + 60, 70)))
        parts.append(
            f'<path d="M {gx + 40} {gy - 1} l -4.5 -7 l 9 0 z" fill="{c_dense}"/>')
    # gate -> generation
    parts.append(_edge(
        f"M {gx + gw} 206 C {gx + gw + 26} 206, {ox - 26} 142, {ox - 8} 142",
        color=palette["ok"], active=gen_active, palette=palette,
        label="đạt ngưỡng" if gen_active else "", label_xy=(ox - 14, 126),
        label_anchor="end"))
    parts.append(_arrow(ox - 1, 142, palette["ok"], gen_active))
    # gate -> fallback
    parts.append(_edge(
        f"M {gx + gw} 254 C {gx + gw + 26} 254, {fx - 26} 328, {fx - 8} 328",
        color=c_fallback, active=fb_active, palette=palette,
        label="dưới ngưỡng" if fb_active else "", label_xy=(fx - 14, 354),
        label_anchor="end"))
    parts.append(_arrow(fx - 1, 328, c_fallback, fb_active))

    # --- decision caption ---------------------------------------------------
    reason = str(trace.get("decision_reason") or "")
    if len(reason) > 118:
        reason = reason[:117] + "…"
    parts.append(
        f'<rect x="24" y="396" width="{_W - 48}" height="30" rx="9" '
        f'fill="{_soft(head_color)}"/>'
    )
    parts.append(
        f'<rect x="24" y="396" width="3.5" height="30" rx="2" fill="{head_color}"/>'
    )
    parts.append(_text(42, 416, reason, size=12, fill=palette["text"], weight=500))

    parts.append(_legend(450, palette))
    parts.append("</svg>")
    return "".join(parts)
