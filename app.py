"""Streamlit chatbot for the UTC student-services RAG pipeline.

The interface has one job beyond answering questions: make the pipeline's
decisions visible. Every answer ships with the sources it actually used, the
retrieval method and score for each one, and a set of graphics drawn from the
same ``trace`` the backend produced — the pipeline diagram with this query's
real numbers, the rank movement RRF caused, the threshold comparison that chose
the branch, and where the time went.

Run:
    streamlit run app.py

``DEMO_MODE`` (sidebar: "Dữ liệu mẫu (demo)") swaps the backend for the fixtures
in ``ui/sample_trace.py``. It exists so the UI can be built and demoed without an
API key, a warm index or the network, and so a live demo has a fallback that
cannot fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="RAG · Quy chế & dịch vụ sinh viên UTC",
    page_icon=":material/schema:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import charts, components, theme  # noqa: E402  (must follow set_page_config)
from ui.pipeline_svg import render_pipeline_svg  # noqa: E402
from ui.sample_trace import (  # noqa: E402
    CORPUS_STATS,
    SAMPLE_QUESTIONS,
    sample_generation,
)

load_dotenv()

# Mirrors src/task9_retrieval_pipeline's own resolution so the slider's default
# and the backend's default can never disagree. The starter ships 0.30, but a
# corpus embedded with multilingual-e5-small needs ~0.853 (see README) — reading
# the env keeps the UI honest without hard-coding either number.
try:
    DEFAULT_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or 0.30)
except ValueError:
    DEFAULT_THRESHOLD = 0.30

# --- backend wiring ----------------------------------------------------------
# The UI must start even when src/ is mid-implementation, so the import is
# guarded and the failure is surfaced as a banner instead of a stack trace.
BACKEND_ERROR: str | None = None
try:  # pragma: no cover - import guard
    from src.task9_retrieval_pipeline import retrieve_with_trace
    from src.task10_generation import generate_with_trace
except Exception as error:  # noqa: BLE001 - any import failure must degrade
    retrieve_with_trace = None  # type: ignore[assignment]
    generate_with_trace = None  # type: ignore[assignment]
    BACKEND_ERROR = f"{type(error).__name__}: {error}"

ROOT = Path(__file__).parent
STANDARDIZED = ROOT / "data" / "standardized"


@st.cache_data(show_spinner=False)
def corpus_counts() -> tuple[int, int]:
    """Count standardized documents on disk — cheap, and real even in live mode."""
    legal = len(list((STANDARDIZED / "legal").glob("*.md"))) if STANDARDIZED.exists() else 0
    news = len(list((STANDARDIZED / "news").glob("*.md"))) if STANDARDIZED.exists() else 0
    return legal, news


@st.cache_data(show_spinner=False)
def chunk_count() -> int | None:
    """Read the indexed chunk count from the shared chunk file.

    Reads ``data/index/chunks.json`` rather than opening ChromaDB: importing the
    index module would pull in sentence-transformers and stall app start by tens
    of seconds just to render one number in the header.
    """
    path = ROOT / "data" / "index" / "chunks.json"
    if not path.exists():
        return None
    try:
        import json

        with path.open(encoding="utf-8") as handle:
            return len(json.load(handle))
    except Exception:  # noqa: BLE001 - a header stat must never break the app
        return None


def _call_generate(
    query: str,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
    history: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the real backend, tolerating a narrower signature than we hope for.

    The frozen contract only promises ``generate_with_trace(query, top_k)``. We
    try to pass the retrieval knobs and the conversation history anyway, and
    fall back through progressively smaller signatures, so the app keeps working
    whichever version of Task 10 is on disk.
    """
    attempts: list[dict[str, Any]] = [
        {"top_k": top_k, "score_threshold": score_threshold,
         "use_reranking": use_reranking, "history": history},
        {"top_k": top_k, "history": history},
        {"top_k": top_k},
    ]
    last: TypeError | None = None
    for kwargs in attempts:
        try:
            return generate_with_trace(query, **kwargs)  # type: ignore[misc]
        except TypeError as error:  # unexpected kwarg -> try a smaller signature
            last = error
    raise last or TypeError("generate_with_trace không gọi được")


def run_turn(
    query: str,
    *,
    demo: bool,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
    scenario: str | None,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    """Produce one assistant turn: result, trace and how we got there."""
    if demo or generate_with_trace is None:
        result, trace = sample_generation(
            query, top_k=top_k, score_threshold=score_threshold,
            use_reranking=use_reranking, scenario=scenario,
        )
        mode = "demo" if demo else "demo-fallback"
        return {"result": result, "trace": trace, "mode": mode, "error": None}

    try:
        result, trace = _call_generate(
            query, top_k, score_threshold, use_reranking, history
        )
        return {"result": result, "trace": trace, "mode": "live", "error": None}
    except Exception as generation_error:  # noqa: BLE001
        # Generation is the fragile half (API key, quota, network). Retrieval
        # usually still works, and a retrieval-only turn is far more useful on
        # stage than a red traceback.
        if retrieve_with_trace is not None:
            try:
                results, trace = retrieve_with_trace(
                    query, top_k=top_k, score_threshold=score_threshold,
                    use_reranking=use_reranking,
                )
                return {
                    "result": {"answer": "", "sources": results,
                               "retrieval_source": trace.get("decision", "hybrid")},
                    "trace": trace,
                    "mode": "retrieval-only",
                    "error": f"{type(generation_error).__name__}: {generation_error}",
                }
            except Exception as retrieval_error:  # noqa: BLE001
                return {"result": None, "trace": None, "mode": "error",
                        "error": f"{type(retrieval_error).__name__}: {retrieval_error}"}
        return {"result": None, "trace": None, "mode": "error",
                "error": f"{type(generation_error).__name__}: {generation_error}"}


# --- sidebar -----------------------------------------------------------------
theme.inject_css()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.05rem;font-weight:700;letter-spacing:-.01em">'
        "Bảng điều khiển pipeline</div>"
        '<div style="font-size:.78rem;color:#8DA0C0;margin-bottom:10px">'
        "Đổi tham số rồi hỏi lại để thấy quyết định thay đổi.</div>",
        unsafe_allow_html=True,
    )

    demo_mode = st.toggle(
        "Dữ liệu mẫu (demo)",
        value=True,
        help="Dùng trace mẫu trong ui/sample_trace.py thay cho backend thật. "
             "Bật để demo khi chưa có API key hoặc chưa index xong.",
    )

    scenario_choice = "Tự động"
    if demo_mode:
        scenario_choice = st.radio(
            "Kịch bản mẫu",
            ("Tự động", "Có căn cứ", "Từ chối an toàn"),
            horizontal=False,
            help="Tự động: đoán theo nội dung câu hỏi. Chọn tay để trình diễn "
                 "đúng nhánh mình muốn.",
        )

    st.divider()
    top_k = st.slider("Số chunk lấy về (top_k)", 3, 10, 5)
    score_threshold = st.slider(
        "Ngưỡng điểm dense", 0.0, 1.0, DEFAULT_THRESHOLD, 0.001,
        format="%.3f",
        help="So sánh với cosine score gốc của dense, không phải điểm RRF. "
             "Dưới ngưỡng thì pipeline chuyển sang fallback vectorless.",
    )
    st.caption(
        f"Mặc định {DEFAULT_THRESHOLD:.3f} đọc từ `SCORE_THRESHOLD`. "
        "Model embedding của corpus này dồn mọi cosine vào dải 0.82–0.92, "
        "nên 0.30 sẽ không bao giờ kích hoạt fallback."
    )
    strategy = st.radio(
        "Chiến lược truy hồi",
        ("Hybrid + RRF", "Chỉ Dense"),
        help="Hybrid gộp bảng xếp hạng của dense và BM25 bằng RRF. "
             "Chỉ Dense tắt bước gộp để so sánh.",
    )
    use_reranking = strategy == "Hybrid + RRF"

    st.divider()
    st.markdown(
        '<div style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;'
        'color:#5C6E8C;font-weight:600;margin-bottom:8px">Mã màu</div>',
        unsafe_allow_html=True,
    )
    st.html(components.color_legend())

    st.divider()
    st.markdown(
        '<div style="font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;'
        'color:#5C6E8C;font-weight:600;margin-bottom:6px">Câu hỏi thử</div>',
        unsafe_allow_html=True,
    )
    for i, question in enumerate(SAMPLE_QUESTIONS):
        if st.button(question, key=f"sample_{i}", width="stretch"):
            st.session_state.pending = question

    if st.session_state.messages:
        st.divider()
        if st.button("Xoá hội thoại", key="clear", width="stretch"):
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

scenario = {
    "Tự động": None, "Có căn cứ": "grounded", "Từ chối an toàn": "refusal",
}.get(scenario_choice)

# --- header ------------------------------------------------------------------
legal_count, news_count = corpus_counts()
if demo_mode:
    legal_count = legal_count or CORPUS_STATS["legal"]
    news_count = news_count or CORPUS_STATS["news"]

indexed = chunk_count()
provider = os.getenv("LLM_PROVIDER", "gemini")
model = os.getenv("LLM_MODEL", "") or "mặc định theo provider"
embedding = os.getenv("EMBEDDING_MODEL", CORPUS_STATS["embedding_model"])

if demo_mode:
    mode_value, mode_dot = "Dữ liệu mẫu", theme.COLORS["hybrid"]
elif BACKEND_ERROR:
    mode_value, mode_dot = "Backend lỗi", theme.COLORS["fallback"]
else:
    mode_value, mode_dot = "Trực tiếp", theme.COLORS["ok"]

st.html(components.header([
    ("Kho tài liệu", f"{legal_count + news_count} tài liệu", theme.COLORS["ok"]),
    ("Phân loại", f"{legal_count} pháp quy · {news_count} tin", None),
    ("Đoạn đã index",
     f"{indexed if indexed is not None else CORPUS_STATS['chunks']} chunk",
     theme.COLORS["ok"] if indexed else None),
    ("Embedding", embedding, theme.COLORS["dense"]),
    ("LLM", f"{provider} · {model}", theme.COLORS["ok"]),
    ("Chế độ", mode_value, mode_dot),
]))

if BACKEND_ERROR and not demo_mode:
    st.html(components.banner(
        f"<b>Không nạp được backend.</b> <code>{BACKEND_ERROR}</code><br>"
        "Giao diện vẫn chạy bằng dữ liệu mẫu. Bật <b>Dữ liệu mẫu (demo)</b> ở "
        "thanh bên để bỏ cảnh báo này.",
        kind="warn",
    ))
elif BACKEND_ERROR and demo_mode:
    st.html(components.banner(
        "Đang chạy bằng <b>dữ liệu mẫu</b>. Mọi số liệu trên biểu đồ lấy từ "
        "<code>ui/sample_trace.py</code>, không phải từ truy hồi thật.",
        kind="info",
    ))


# --- rendering ---------------------------------------------------------------
def render_svg(svg: str) -> None:
    """Render the pipeline diagram.

    Verified on Streamlit 1.64: ``st.html`` runs the payload through a sanitiser
    configured for HTML only, which deletes the entire ``<svg>`` subtree — the
    diagram silently disappears. ``st.markdown(unsafe_allow_html=True)`` keeps
    both the SVG and its SMIL ``<animate>`` elements, so that is the path used
    here. Do not "simplify" this back to ``st.html``.
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_analytics(trace: dict[str, Any], turn: int, *, compact: bool = False) -> None:
    """The four explanatory graphics for one turn."""
    if not trace:
        return
    if compact:
        render_svg(render_pipeline_svg(trace))
        return

    tab_svg, tab_rank, tab_score, tab_time, tab_raw = st.tabs([
        "Sơ đồ pipeline", "RRF đã đổi thứ hạng thế nào", "So sánh điểm",
        "Ngưỡng & thời gian", "Trace thô",
    ])
    with tab_svg:
        render_svg(render_pipeline_svg(trace))
    with tab_rank:
        st.plotly_chart(
            charts.rank_movement_slope(trace), config=charts.PLOTLY_CONFIG,
            theme=None, key=f"slope_{turn}",
        )
        st.caption(
            "Mỗi đường là một chunk. Cột nào không có chấm nghĩa là retriever đó "
            "không tìm thấy chunk này — RRF vẫn giữ nó lại nếu retriever còn lại "
            "xếp hạng đủ cao."
        )
    with tab_score:
        st.plotly_chart(
            charts.score_comparison_bars(trace), config=charts.PLOTLY_CONFIG,
            theme=None, key=f"scores_{turn}",
        )
        st.caption(
            "BM25 và RRF được chuẩn hoá về 0–1 để vẽ chung; giá trị gốc hiện khi "
            "rê chuột. Chỉ cột dense mới cùng thang với ngưỡng."
        )
    with tab_time:
        left, right = st.columns([1, 1.35], gap="medium")
        with left:
            st.plotly_chart(
                charts.threshold_gauge(trace), config=charts.PLOTLY_CONFIG,
                theme=None, key=f"gauge_{turn}",
            )
        with right:
            st.plotly_chart(
                charts.latency_breakdown(trace), config=charts.PLOTLY_CONFIG,
                theme=None, key=f"latency_{turn}",
            )
            reorder = components.reorder_table(trace)
            if reorder:
                st.html(reorder)
    with tab_raw:
        st.json(
            {k: v for k, v in trace.items() if k != "stages"}, expanded=False
        )
        for stage in trace.get("stages") or []:
            st.markdown(
                f"**{stage.get('label', stage.get('name'))}** — "
                f"{stage.get('count', 0)} kết quả, {stage.get('elapsed_ms', 0)} ms"
            )


def render_turn(message: dict[str, Any], turn: int, *, latest: bool) -> None:
    """Render one assistant message with its sources and graphics."""
    trace: dict[str, Any] = message.get("trace") or {}
    result: dict[str, Any] = message.get("result") or {}
    mode = message.get("mode")
    query = message.get("query", "")

    if mode == "error":
        st.html(components.banner(
            f"<b>Pipeline lỗi.</b> <code>{message.get('error')}</code>", kind="warn",
        ))
        return

    if mode == "retrieval-only":
        st.html(components.banner(
            "<b>Chỉ chạy được phần truy hồi.</b> Sinh câu trả lời thất bại: "
            f"<code>{message.get('error')}</code> — vẫn hiển thị đầy đủ nguồn và "
            "phân tích bên dưới.",
            kind="warn",
        ))

    sources: list[dict[str, Any]] = result.get("sources") or []
    refused = (
        result.get("retrieval_source") == "none"
        or bool(trace.get("refused"))
        or not sources
    )

    # Numeric read-out first: it is the same four numbers the graphics expand on.
    stages = {s.get("name"): s for s in (trace.get("stages") or [])}
    decision = str(trace.get("decision") or result.get("retrieval_source") or "—")
    st.html(components.metric_tiles([
        ("Nhánh đã chọn", decision,
         "vượt ngưỡng" if decision == "hybrid" else "dưới ngưỡng",
         theme.COLORS["hybrid"] if decision == "hybrid" else theme.COLORS["fallback"]),
        ("Dense tốt nhất", f"{float(trace.get('best_dense_score') or 0):.3f}",
         f"ngưỡng {float(trace.get('score_threshold') or 0):.3f}",
         theme.COLORS["dense"]),
        ("Ứng viên dense / BM25",
         f"{stages.get('dense', {}).get('count', 0)} / "
         f"{stages.get('bm25', {}).get('count', 0)}",
         f"RRF k = {stages.get('rrf', {}).get('k', 60)}", theme.COLORS["bm25"]),
        ("Nguồn đã dùng", str(len(sources)),
         f"top_k = {trace.get('top_k', '—')}", theme.COLORS["ok"]),
        ("Thời gian", f"{float(trace.get('elapsed_ms') or 0):.0f} ms",
         f"LLM {float(trace.get('llm_elapsed_ms') or 0):.0f} ms", theme.COLORS["ok"]),
    ]))

    if refused:
        st.html(components.refusal_card(
            trace,
            result.get("answer")
            or "Không đủ căn cứ trong tài liệu để trả lời câu hỏi này.",
        ))
        weak = (
            (stages.get("rrf", {}).get("results")
             or stages.get("dense", {}).get("results") or [])[: trace.get("top_k", 5)]
        )
        if weak:
            st.markdown(
                '<div style="font-size:.78rem;color:#8DA0C0;margin:6px 0 6px">'
                "Những đoạn gần nhất mà pipeline tìm được nhưng <b>không đạt ngưỡng</b> "
                "— hiển thị để kiểm chứng, không dùng làm căn cứ trả lời:</div>",
                unsafe_allow_html=True,
            )
            st.html(components.source_list(
                weak, turn=turn, query=query, dimmed=True
            ))
    elif mode == "retrieval-only":
        st.html(components.source_list(sources, turn=turn, query=query))
    else:
        st.html(components.answer_card(
            result.get("answer", ""), trace.get("citations") or [], turn, len(sources)
        ))
        st.markdown(
            '<div style="font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;'
            'color:#5C6E8C;font-weight:600;margin:14px 0 8px">'
            "Nguồn đã dùng · bấm số [n] trong câu trả lời để nhảy tới</div>",
            unsafe_allow_html=True,
        )
        st.html(components.source_list(sources, turn=turn, query=query))

    st.write("")
    if latest or st.session_state.get(f"expand_{turn}"):
        render_analytics(trace, turn)
    else:
        with st.expander("Phân tích pipeline của lượt này"):
            render_analytics(trace, turn, compact=True)
            st.button(
                "Mở đầy đủ biểu đồ", key=f"expand_btn_{turn}",
                on_click=lambda t=turn: st.session_state.__setitem__(f"expand_{t}", True),
            )


last_index = len(st.session_state.messages) - 1
for index, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.html(components.user_bubble(message["content"]))
    else:
        with st.chat_message("assistant", avatar=":material/hub:"):
            render_turn(message, index, latest=index == last_index)

if not st.session_state.messages:
    st.html(components.banner(
        "Hỏi một câu về quy chế đào tạo hoặc dịch vụ sinh viên của trường. "
        "Mỗi câu trả lời sẽ kèm sơ đồ pipeline, bảng thay đổi thứ hạng do RRF "
        "và toàn bộ nguồn đã dùng. Thử nhanh bằng các câu hỏi mẫu ở thanh bên.",
        kind="info",
    ))

# --- input -------------------------------------------------------------------
typed = st.chat_input("Nhập câu hỏi về quy chế hoặc dịch vụ sinh viên…")
query = typed or st.session_state.pending
if query:
    st.session_state.pending = None
    history = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in st.session_state.messages
        if m.get("content")
    ]
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar=":material/hub:"):
        with st.spinner("Đang truy hồi và đối chiếu nguồn…"):
            turn = run_turn(
                query,
                demo=demo_mode,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
                scenario=scenario,
                history=history,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": (turn.get("result") or {}).get("answer", ""),
        "query": query,
        **turn,
    })
    st.rerun()
