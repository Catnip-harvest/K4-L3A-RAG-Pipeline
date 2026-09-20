"""
Task 10 — Generation có citation.

Hai điểm thiết kế đáng nói khi demo.

**1. Số citation gắn với thứ hạng, không gắn với vị trí trong prompt.**
``reorder_for_llm`` xáo lại chunks để chống "lost in the middle" — model chú ý
kém ở giữa context dài, nên chunk mạnh nhất phải nằm ở đầu và cuối. Nhưng
contract lại bắt ``sources`` sắp xếp theo score giảm dần. Nếu đánh số citation
theo vị trí vật lý trong prompt thì ``[2]`` trong câu trả lời sẽ trỏ sang
``sources[1]`` khác hẳn. Vì vậy số citation được tính từ **thứ hạng score** và
giữ nguyên dù chunk nằm ở đâu trong prompt.

**2. Safe refusal là một nhánh có chủ đích, không phải lỗi.**
Không retrieve được gì, hoặc provider chết, thì trả lời từ chối kèm
``retrieval_source="none"``. Chatbot quy chế mà đoán bừa thì nguy hiểm hơn là
im lặng.
"""

from __future__ import annotations

import os
import re
import time

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve_with_trace


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
}.get(LLM_PROVIDER, "gemini-2.5-flash")

REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu quy định và dịch vụ sinh viên của Trường Đại học Giao thông Vận tải.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên CONTEXT được cung cấp. Không dùng kiến thức bên ngoài.
- Mỗi khẳng định phải kèm citation dạng [số] đúng theo số Document trong context.
- Nếu context không chứa đủ căn cứ, trả lời đúng một câu: "Tôi không thể xác minh thông tin này từ nguồn hiện có." và không bịa thêm.
- Nếu hai tài liệu mâu thuẫn nhau, nêu rõ cả hai và chỉ ra tài liệu nào mới hơn.
- Trả lời bằng tiếng Việt, ngắn gọn, đi thẳng vào câu hỏi."""

_CITATION = re.compile(r"\[(\d+)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context.

    Không sửa danh sách đầu vào — Task 9 và UI vẫn cần thứ tự theo score.
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def citation_numbers(chunks: list[dict]) -> dict[str, int]:
    """Gán số citation theo thứ hạng score giảm dần, ổn định với mọi thứ tự."""
    ranked = sorted(
        chunks,
        key=lambda chunk: (-float(chunk.get("score", 0.0)), chunk.get("id", "")),
    )
    return {chunk["id"]: index for index, chunk in enumerate(ranked, start=1)}


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    numbers = citation_numbers(chunks)
    parts = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        parts.append(
            f"[Document {numbers[chunk['id']]} | Title: {metadata.get('title', '')} | "
            f"Source: {metadata.get('source', '')}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình trong .env."""
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return (response.text or "").strip()

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    if LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    raise ValueError(f"LLM_PROVIDER không hỗ trợ: {LLM_PROVIDER}")


def _build_user_message(query: str, context: str, history: list[dict] | None) -> str:
    """Ghép context, lịch sử hội thoại và câu hỏi thành một prompt."""
    blocks = [f"Context:\n{context}"]
    if history:
        # Chỉ giữ vài lượt gần nhất: đủ để giải nghĩa "nó", "cái đó" trong câu
        # hỏi nối tiếp, mà không đẩy context tài liệu ra khỏi vùng model chú ý.
        recent = history[-4:]
        turns = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)
        blocks.append(f"Hội thoại trước đó (chỉ dùng để hiểu câu hỏi, không dùng làm nguồn):\n{turns}")
    blocks.append(f"Question: {query}")
    return "\n\n".join(blocks)


def _empty_trace(query: str, top_k: int, reason: str) -> dict:
    # decision="none" nằm ngoài enum "hybrid"|"pageindex" của retrieval, và đó là
    # cố ý: nó phản chiếu GenerationResult.retrieval_source, vốn có nhận "none".
    # Nghĩa là "không đường nào chạy" chứ không phải một đường thứ ba.
    return {
        "query": query,
        "top_k": top_k,
        "stages": [],
        "decision": "none",
        "decision_reason": reason,
        "rank_movement": [],
        "final_ids": [],
        "refused": True,
        "refusal_stage": "empty",
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "llm_elapsed_ms": 0.0,
        "context_chars": 0,
        "prompt_chars": 0,
        "reorder_map": [],
        "citations": [],
    }


def generate_with_trace(
    query: str,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
    score_threshold: float | None = None,
    use_reranking: bool = True,
    balanced: bool = False,
) -> tuple[dict, dict]:
    """Như ``generate_with_citation`` nhưng kèm bản ghi từng bước cho UI.

    ``history`` là bộ nhớ hội thoại cho câu hỏi nối tiếp; nó chỉ giúp model hiểu
    đại từ, không bao giờ được dùng làm nguồn dữ kiện.

    ``use_reranking`` là công tắc A/B của bài đánh giá: tắt đi thì pipeline chạy
    dense-only (Config A), bật thì hybrid + RRF (Config B). Mọi thứ còn lại —
    prompt, generator, top_k, golden dataset — phải giữ nguyên giữa hai lần chạy,
    nếu không thì chênh lệch metric không nói lên điều gì về RRF.
    """
    retrieval_kwargs = {"top_k": top_k, "use_reranking": use_reranking, "balanced": balanced}
    if score_threshold is not None:
        retrieval_kwargs["score_threshold"] = score_threshold
    chunks, trace = retrieve_with_trace(query, **retrieval_kwargs)

    if not chunks:
        trace.update(_empty_trace(query, top_k, "Không retrieve được chunk nào"))
        return {"answer": REFUSAL, "sources": [], "retrieval_source": "none"}, trace

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = _build_user_message(query, context, history)

    started = time.perf_counter()
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        llm_ms = round((time.perf_counter() - started) * 1000, 1)
        refused = not answer or REFUSAL[:30] in answer
    except Exception as error:
        # Provider chết thì từ chối an toàn, không dựng câu trả lời không nguồn.
        #
        # QUAN TRỌNG: lỗi provider KHÔNG phải là "từ chối". Người dùng nhìn thấy
        # cùng một câu, nhưng hai thứ này hoàn toàn khác nhau khi đo:
        #   - refused  = model đọc context rồi kết luận không đủ căn cứ (hành vi đúng)
        #   - errored  = chưa bao giờ gọi được model (mất dữ liệu đo)
        # Gộp chung hai trạng thái làm hỏng bài đánh giá: một đợt rate limit 429
        # sẽ hiện ra thành "config này từ chối nhiều hơn", tức là kết luận sai
        # hoàn toàn về chất lượng retrieval.
        llm_ms = round((time.perf_counter() - started) * 1000, 1)
        trace.update(
            {
                "refused": False,
                "errored": True,
                "refusal_stage": "provider_error",
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "llm_elapsed_ms": llm_ms,
                "llm_error": f"{type(error).__name__}: {error}"[:300],
                "context_chars": len(context),
                "prompt_chars": len(user_message),
                "reorder_map": [],
                "citations": [],
            }
        )
        return (
            {"answer": REFUSAL, "sources": chunks, "retrieval_source": "none"},
            trace,
        )

    numbers = citation_numbers(chunks)
    by_number = {number: chunk for chunk, number in ((c, numbers[c["id"]]) for c in chunks)}
    used = sorted({int(marker) for marker in _CITATION.findall(answer)})
    citations = [
        {
            "marker": f"[{number}]",
            "source_id": by_number[number]["id"] if number in by_number else "",
            "title": (by_number[number].get("metadata") or {}).get("title", "")
            if number in by_number
            else "",
            "matched": number in by_number,
        }
        for number in used
    ]

    positions = {chunk["id"]: index for index, chunk in enumerate(chunks, start=1)}
    trace.update(
        {
            "refused": refused,
            "errored": False,
            # Ba lý do từ chối rất khác nhau và UI phải phân biệt được: model
            # đọc context rồi quyết định không đủ căn cứ ("generation"), không
            # retrieve được gì ("empty"), hay chưa từng gọi được model
            # ("provider_error" — đây là mất dữ liệu, không phải hành vi đúng).
            "refusal_stage": "generation" if refused else None,
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "llm_elapsed_ms": llm_ms,
            "context_chars": len(context),
            "prompt_chars": len(user_message),
            "reorder_map": [
                {"id": chunk["id"], "from_rank": positions[chunk["id"]], "to_rank": index}
                for index, chunk in enumerate(reordered, start=1)
            ],
            "citations": citations,
        }
    )

    result = {
        "answer": answer or REFUSAL,
        "sources": chunks,
        "retrieval_source": "none" if refused else chunks[0]["retrieval_method"],
    }
    # retrieval_source chỉ nhận hybrid | pageindex | none.
    if result["retrieval_source"] not in {"hybrid", "pageindex", "none"}:
        result["retrieval_source"] = "hybrid"
    return result, trace


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    result, _ = generate_with_trace(query, top_k=top_k)
    return result


if __name__ == "__main__":
    output = generate_with_citation("Ai quản lý ký túc xá của trường?")
    print(output["answer"])
    print(f"\nretrieval_source = {output['retrieval_source']}")
    for source in output["sources"]:
        print(f"  {source['score']:.4f}  {source['id']}")
