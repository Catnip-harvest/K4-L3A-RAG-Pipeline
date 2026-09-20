"""Colour tokens and global CSS for the RAG demo UI.

Design rule that drives everything in this file: the audience must learn the
colour code in five seconds. Exactly four saturated colours are allowed in the
whole product, and each one means one stage of the pipeline:

    violet  -> dense / vector retrieval
    cyan    -> BM25 / keyword retrieval
    amber   -> hybrid (RRF fusion) and the path that was actually taken
    rose    -> the vectorless fallback / refusal path

Everything else (document type pills, chrome, borders) is deliberately
desaturated so the four signal colours never compete with decoration. The same
tokens are consumed by the SVG diagram, the Plotly figures and the badges, so a
colour learned in one graphic reads the same in the next.
"""

from __future__ import annotations

from string import Template
from typing import Final

import streamlit as st


# --- Palette -----------------------------------------------------------------
# Kept as a plain dict (not a dataclass) because it is interpolated into CSS,
# SVG and Plotly layouts as strings far more often than it is attribute-accessed.
COLORS: Final[dict[str, str]] = {
    # canvas
    "bg": "#080C16",
    "bg_soft": "#0C1220",
    "surface": "#111A2C",
    "surface_2": "#18243B",
    "surface_3": "#1E2C46",
    "border": "#243350",
    "border_soft": "#1B2740",
    # type
    "text": "#E9F0FC",
    "text_dim": "#B9C7DE",
    "muted": "#8DA0C0",
    "faint": "#5C6E8C",
    # the four signal colours
    "dense": "#7C5CFF",
    "dense_soft": "#2A2153",
    "bm25": "#22D3EE",
    "bm25_soft": "#12333F",
    "hybrid": "#FFB020",
    "hybrid_soft": "#3B2A0E",
    "fallback": "#FB7185",
    "fallback_soft": "#3A1B26",
    # status
    "ok": "#34D399",
    "warn": "#FBBF24",
    "danger": "#F87171",
}

FONT_STACK: Final[str] = (
    '"Be Vietnam Pro", "Inter", -apple-system, BlinkMacSystemFont, '
    '"Segoe UI", Roboto, sans-serif'
)
MONO_STACK: Final[str] = '"JetBrains Mono", "SFMono-Regular", Consolas, monospace'

# Be Vietnam Pro is chosen over Inter because its diacritic marks are drawn for
# Vietnamese specifically - stacked tone marks on chu hoa do not collide, which
# matters for a UI that is entirely in Vietnamese.
FONT_IMPORT: Final[str] = (
    "https://fonts.googleapis.com/css2"
    "?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap"
)

METHOD_COLORS: Final[dict[str, str]] = {
    "dense": COLORS["dense"],
    "bm25": COLORS["bm25"],
    "hybrid": COLORS["hybrid"],
    "pageindex": COLORS["fallback"],
}

METHOD_LABELS: Final[dict[str, str]] = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid RRF",
    "pageindex": "PageIndex",
}

DOC_TYPE_LABELS: Final[dict[str, str]] = {
    "legal": "PHÁP QUY",
    "news": "TIN TỨC",
}


_CSS = Template(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="$font_url">
<style>
@import url("$font_url");

:root {
  --rag-bg: $bg;
  --rag-surface: $surface;
  --rag-surface-2: $surface_2;
  --rag-surface-3: $surface_3;
  --rag-border: $border;
  --rag-border-soft: $border_soft;
  --rag-text: $text;
  --rag-text-dim: $text_dim;
  --rag-muted: $muted;
  --rag-faint: $faint;
  --rag-dense: $dense;
  --rag-bm25: $bm25;
  --rag-hybrid: $hybrid;
  --rag-fallback: $fallback;
  --rag-ok: $ok;
  --rag-warn: $warn;
  --rag-font: $font_stack;
  --rag-mono: $mono_stack;
}

html, body, [class*="st-"], button, input, textarea, select {
  font-family: var(--rag-font) !important;
  -webkit-font-smoothing: antialiased;
}

/* Streamlit renders its icons as ligatures in Material Symbols. The blanket
   font override above also hits those spans, which makes the icon show its
   ligature NAME as literal text ("keyboard_double_arrow_left" in the sidebar
   collapse button). Hand the icon font back. */
[data-testid="stIconMaterial"],
span[class*="material-symbols"],
.material-symbols-rounded {
  font-family: "Material Symbols Rounded" !important;
}

.stApp {
  background:
    radial-gradient(1100px 520px at 12% -12%, rgba(124, 92, 255, .16), transparent 62%),
    radial-gradient(900px 480px at 95% 0%, rgba(34, 211, 238, .10), transparent 58%),
    $bg;
}

/* Tighten Streamlit's very generous default chrome so the graphics get room. */
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 1.1rem !important;
  padding-bottom: 5.5rem !important;
  max-width: 1320px;
}
[data-testid="stHeader"] { background: transparent; height: 0rem; }
[data-testid="stToolbar"] { right: .6rem; }
footer { display: none; }

h1, h2, h3, h4 { letter-spacing: -.02em; font-weight: 700; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, $surface 0%, $bg_soft 100%);
  border-right: 1px solid var(--rag-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* ---------- header ---------- */
.rag-header {
  border: 1px solid var(--rag-border);
  border-radius: 20px;
  padding: 18px 22px;
  margin-bottom: 14px;
  background:
    linear-gradient(135deg, rgba(124, 92, 255, .16), rgba(34, 211, 238, .07) 48%, rgba(255, 176, 32, .07)),
    var(--rag-surface);
  box-shadow: 0 18px 46px -26px rgba(0, 0, 0, .95), inset 0 1px 0 rgba(255, 255, 255, .05);
}
.rag-header__row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.rag-mark {
  width: 42px; height: 42px; border-radius: 13px; flex: none;
  background: linear-gradient(135deg, $dense, $bm25);
  display: grid; place-items: center;
  font-weight: 800; font-size: 15px; color: #08101E; letter-spacing: -.04em;
  box-shadow: 0 10px 24px -10px rgba(124, 92, 255, .9);
}
.rag-title { font-size: 1.62rem; font-weight: 800; line-height: 1.1; margin: 0;
  background: linear-gradient(92deg, #FFFFFF 0%, #C9D6F2 55%, #9FB4DC 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.rag-sub { color: var(--rag-muted); font-size: .88rem; margin: 3px 0 0; }

/* ---------- status strip ---------- */
.rag-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.rag-strip__item {
  flex: 1 1 165px; min-width: 150px;
  border: 1px solid var(--rag-border-soft); border-radius: 13px;
  padding: 9px 12px; background: rgba(8, 12, 22, .55);
}
.rag-strip__k { font-size: .64rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--rag-faint); font-weight: 600; }
.rag-strip__v { font-size: .92rem; font-weight: 600; color: var(--rag-text);
  margin-top: 2px; font-family: var(--rag-mono); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.rag-strip__v .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle; }

/* ---------- generic card ---------- */
.rag-card {
  border: 1px solid var(--rag-border); border-radius: 16px;
  background: var(--rag-surface); padding: 15px 17px;
  box-shadow: 0 14px 34px -24px rgba(0, 0, 0, .9);
}

/* ---------- answer ---------- */
.rag-answer {
  border: 1px solid var(--rag-border); border-left: 3px solid $hybrid;
  border-radius: 14px; padding: 14px 18px; margin: 2px 0 12px;
  background: linear-gradient(180deg, rgba(255, 176, 32, .05), rgba(17, 26, 44, .0)), var(--rag-surface);
  line-height: 1.68; font-size: .97rem; color: var(--rag-text);
}
.rag-answer p { margin: 0 0 .62em; }
.rag-answer p:last-child { margin-bottom: 0; }
.rag-answer ul, .rag-answer ol { margin: .3em 0 .7em 1.1em; }
.rag-answer li { margin: .22em 0; }

/* ---------- citation chips ---------- */
a.rag-cite {
  display: inline-block; text-decoration: none !important;
  min-width: 19px; height: 19px; line-height: 18px; padding: 0 5px;
  margin: 0 2px; border-radius: 6px; text-align: center;
  font-family: var(--rag-mono); font-size: .7rem; font-weight: 600;
  color: $hybrid !important; background: rgba(255, 176, 32, .13);
  border: 1px solid rgba(255, 176, 32, .42);
  transform: translateY(-1px); transition: all .14s ease;
}
a.rag-cite:hover { background: $hybrid; color: #1A1206 !important;
  box-shadow: 0 0 0 3px rgba(255, 176, 32, .18); }
a.rag-cite.is-unmatched { color: $fallback !important;
  background: rgba(251, 113, 133, .12); border-color: rgba(251, 113, 133, .45); }

/* ---------- source cards ---------- */
.rag-sources { display: flex; flex-direction: column; gap: 9px; }
.rag-src {
  position: relative; scroll-margin-top: 90px;
  border: 1px solid var(--rag-border); border-radius: 14px;
  background: var(--rag-surface); padding: 12px 14px 12px 15px;
  transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
}
.rag-src::before {
  content: ""; position: absolute; left: 0; top: 12px; bottom: 12px; width: 3px;
  border-radius: 0 3px 3px 0; background: var(--rag-accent, $muted);
}
.rag-src:hover { border-color: var(--rag-accent, $border); }
/* :target is the whole citation-jump mechanism - clicking [n] in the answer sets
   the hash, and the matching card lights up without a single line of JS. */
.rag-src:target {
  border-color: $hybrid;
  background: linear-gradient(180deg, rgba(255, 176, 32, .11), rgba(17, 26, 44, 0)), var(--rag-surface-2);
  box-shadow: 0 0 0 1px rgba(255, 176, 32, .5), 0 0 34px -8px rgba(255, 176, 32, .5);
  animation: rag-pulse 1.5s ease-out 1;
}
@keyframes rag-pulse {
  0% { box-shadow: 0 0 0 1px rgba(255, 176, 32, .9), 0 0 0 10px rgba(255, 176, 32, .22); }
  100% { box-shadow: 0 0 0 1px rgba(255, 176, 32, .5), 0 0 34px -8px rgba(255, 176, 32, .5); }
}
.rag-src__head { display: flex; align-items: flex-start; gap: 10px; }
.rag-src__n {
  flex: none; width: 24px; height: 24px; border-radius: 8px;
  display: grid; place-items: center; font-family: var(--rag-mono);
  font-size: .74rem; font-weight: 700; color: #0A0F1C;
  background: var(--rag-accent, $muted);
}
.rag-src__title { font-weight: 650; font-size: .93rem; color: var(--rag-text);
  line-height: 1.32; margin: 1px 0 0; }
.rag-src__meta { display: flex; align-items: center; gap: 7px; flex-wrap: wrap;
  margin-top: 7px; }
.rag-src__file { font-family: var(--rag-mono); font-size: .7rem; color: var(--rag-faint); }
.rag-src__score { margin-left: auto; text-align: right; flex: none; }
.rag-src__score b { font-family: var(--rag-mono); font-size: 1.0rem; font-weight: 600;
  color: var(--rag-accent, $text); display: block; line-height: 1; }
.rag-src__score span { font-size: .6rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--rag-faint); }
.rag-src__bar { height: 3px; border-radius: 3px; margin-top: 9px;
  background: var(--rag-border-soft); overflow: hidden; }
.rag-src__bar i { display: block; height: 100%; border-radius: 3px;
  background: var(--rag-accent, $muted); }

/* ---------- pills ---------- */
.rag-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 9px; border-radius: 999px;
  font-size: .66rem; font-weight: 600; letter-spacing: .04em;
  border: 1px solid currentColor; line-height: 1.7; white-space: nowrap;
}
.rag-pill .d { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.rag-pill--type { color: var(--rag-faint); border-color: var(--rag-border);
  background: rgba(255, 255, 255, .025); letter-spacing: .1em; }

/* ---------- chunk text ---------- */
.rag-chunk { font-size: .855rem; line-height: 1.72; color: var(--rag-text-dim);
  white-space: pre-wrap; }
.rag-chunk mark {
  background: rgba(34, 211, 238, .2); color: #CFFAFE; padding: 0 3px;
  border-radius: 4px; border-bottom: 1px solid rgba(34, 211, 238, .55);
}

/* ---------- refusal ---------- */
.rag-refusal {
  border: 1px solid rgba(251, 113, 133, .38); border-left: 3px solid $fallback;
  border-radius: 14px; padding: 15px 18px; margin-bottom: 12px;
  background: linear-gradient(180deg, rgba(251, 113, 133, .08), rgba(17, 26, 44, 0)), var(--rag-surface);
}
.rag-refusal__h { display: flex; align-items: center; gap: 9px;
  font-weight: 700; color: #FFD9DF; font-size: .98rem; }
.rag-refusal__h .ic { width: 22px; height: 22px; border-radius: 7px; flex: none;
  display: grid; place-items: center; background: rgba(251, 113, 133, .17);
  color: $fallback; font-weight: 800; font-size: .8rem; }
.rag-refusal__b { color: var(--rag-text-dim); font-size: .9rem; line-height: 1.66;
  margin-top: 8px; }
.rag-refusal__n { margin-top: 11px; padding-top: 11px;
  border-top: 1px dashed rgba(251, 113, 133, .28);
  font-family: var(--rag-mono); font-size: .78rem; color: var(--rag-muted); }

/* ---------- banner ---------- */
.rag-banner { border-radius: 13px; padding: 11px 15px; font-size: .855rem;
  border: 1px solid var(--rag-border); background: var(--rag-surface);
  display: flex; gap: 10px; align-items: flex-start; line-height: 1.55; }
.rag-banner b { font-weight: 650; }
.rag-banner--warn { border-color: rgba(251, 191, 36, .42);
  background: linear-gradient(180deg, rgba(251, 191, 36, .09), rgba(17, 26, 44, 0)), var(--rag-surface); }
.rag-banner--info { border-color: rgba(34, 211, 238, .34);
  background: linear-gradient(180deg, rgba(34, 211, 238, .07), rgba(17, 26, 44, 0)), var(--rag-surface); }
.rag-banner code { font-family: var(--rag-mono); font-size: .8rem;
  background: rgba(0, 0, 0, .35); padding: 1px 5px; border-radius: 5px; }

/* ---------- metric tiles ---------- */
.rag-tiles { display: flex; gap: 9px; flex-wrap: wrap; margin: 2px 0 6px; }
.rag-tile { flex: 1 1 130px; min-width: 118px; border-radius: 13px; padding: 10px 13px;
  border: 1px solid var(--rag-border); background: var(--rag-surface);
  border-top: 2px solid var(--rag-accent, $border); }
.rag-tile__k { font-size: .62rem; letter-spacing: .11em; text-transform: uppercase;
  color: var(--rag-faint); font-weight: 600; }
.rag-tile__v { font-family: var(--rag-mono); font-size: 1.22rem; font-weight: 600;
  color: var(--rag-text); line-height: 1.25; margin-top: 3px; }
.rag-tile__s { font-size: .68rem; color: var(--rag-muted); }

/* ---------- legend ---------- */
.rag-legend { display: flex; flex-direction: column; gap: 6px; }
.rag-legend__i { display: flex; align-items: center; gap: 9px; font-size: .78rem;
  color: var(--rag-text-dim); }
.rag-legend__i i { width: 12px; height: 4px; border-radius: 3px; flex: none; }
.rag-legend__i em { font-style: normal; color: var(--rag-faint); font-size: .72rem; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
  background: transparent; padding: .25rem 0 .1rem; gap: .75rem;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { border: 1px solid var(--rag-border); }
.rag-userq {
  display: inline-block; border-radius: 14px 14px 14px 4px; padding: 9px 15px;
  background: linear-gradient(135deg, rgba(124, 92, 255, .22), rgba(34, 211, 238, .12));
  border: 1px solid rgba(124, 92, 255, .38); color: var(--rag-text);
  font-size: .95rem; font-weight: 500; line-height: 1.5;
}
[data-testid="stChatInput"] {
  border: 1px solid var(--rag-border) !important; border-radius: 15px !important;
  background: var(--rag-surface) !important;
}
[data-testid="stBottomBlockContainer"] { background: transparent; }

/* ---------- tabs / expander ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 3px; background: transparent;
  border-bottom: 1px solid var(--rag-border); }
[data-testid="stTabs"] [data-baseweb="tab"] { height: 38px; padding: 0 14px;
  font-size: .85rem; font-weight: 550; color: var(--rag-muted); }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--rag-text) !important; }
[data-testid="stExpander"] { border: 1px solid var(--rag-border-soft) !important;
  border-radius: 12px !important; background: rgba(8, 12, 22, .45) !important; }
[data-testid="stExpander"] summary { font-size: .82rem; color: var(--rag-muted); }

hr { border-color: var(--rag-border-soft) !important; margin: .8rem 0 !important; }

/* thin scrollbars so wide graphics do not gain a chunky grey rail */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: $surface_3; border-radius: 9px; }
::-webkit-scrollbar-thumb:hover { background: $border; }

/* never let a wide figure or long token push the page sideways */
.rag-answer, .rag-src, .rag-card { overflow-wrap: anywhere; }
</style>
"""
)


def inject_css() -> None:
    """Inject fonts and the global stylesheet once per rerun.

    Uses ``st.html`` rather than ``st.markdown`` because the payload is pure
    CSS: ``st.html`` skips markdown parsing entirely, so nothing in the
    stylesheet can be reinterpreted as markdown syntax.
    """
    st.html(_CSS.substitute(font_url=FONT_IMPORT, font_stack=FONT_STACK,
                            mono_stack=MONO_STACK, **COLORS))


def method_color(method: str) -> str:
    """Colour for a ``retrieval_method`` value, falling back to muted grey."""
    return METHOD_COLORS.get(method, COLORS["muted"])


def method_label(method: str) -> str:
    """Human label for a ``retrieval_method`` value."""
    return METHOD_LABELS.get(method, method or "-")
