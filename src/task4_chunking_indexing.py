"""
Task 4 — Chunking, embedding và indexing.

Đọc Markdown trong data/standardized/, chia chunk, embed bằng một provider duy
nhất và upsert vào ChromaDB (cosine distance).

Ba quyết định cần giải thích khi demo:

1. **Prefix bất đối xứng.** Model mặc định là ``intfloat/multilingual-e5-small``.
   Họ E5 được huấn luyện với hai prefix khác nhau: ``passage: `` cho tài liệu và
   ``query: `` cho câu hỏi. Contract bắt buộc Task 4 và Task 5 dùng chung
   ``embed_texts()``, nên prefix KHÔNG nằm trong hàm đó — hàm chỉ là bộ mã hoá
   thô. Người gọi tự gắn prefix qua ``as_passage()`` / ``as_query()``. Nhờ vậy
   ``embed_texts(texts)`` giữ đúng một tham số vị trí và test có thể monkeypatch.
   Nếu đổi sang model không dùng prefix (ví dụ BGE-M3) thì hai helper này tự
   động trả về nguyên văn.

2. **ID ổn định.** ``<đường dẫn tương đối>::chunk-<index>``. Chạy lại pipeline
   sinh đúng ID cũ nên ``upsert`` ghi đè thay vì nhân bản dữ liệu.

3. **Corpus dùng chung.** Chunks được ghi ra ``data/index/chunks.json`` để Task 6
   (BM25) đọc đúng tập chunk mà Task 5 đã index. Nếu hai bên tự chunk riêng thì
   RRF sẽ gộp hai bảng xếp hạng trỏ tới hai tập ID khác nhau và kết quả vô nghĩa.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
CHUNK_CACHE = Path(__file__).parent.parent / "data" / "index" / "chunks.json"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small").strip()
EMBEDDING_DIM = 384

COLLECTION_NAME = "rag_documents"

# Chỉ họ E5 cần prefix; các model khác coi prefix là nhiễu.
_NEEDS_PREFIX = "e5" in EMBEDDING_MODEL.lower()

_model_cache: dict[str, Any] = {}


def as_passage(text: str) -> str:
    """Gắn prefix cho văn bản được index."""
    return f"passage: {text}" if _NEEDS_PREFIX else text


def as_query(text: str) -> str:
    """Gắn prefix cho câu hỏi của người dùng."""
    return f"query: {text}" if _NEEDS_PREFIX else text


def _sentence_transformer():
    """Nạp model một lần rồi giữ lại — mỗi lần nạp tốn vài giây trên CPU."""
    if "st" not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache["st"] = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache["st"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Mã hoá danh sách văn bản thành vector.

    Đây là bộ mã hoá thô dùng chung cho cả index (Task 4) và truy vấn (Task 5).
    Không tự thêm prefix — xem docstring của module.
    """
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "sentence_transformers":
        model = _sentence_transformer()
        vectors = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        return [list(item.values) for item in response.embeddings]

    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown trong data/standardized/ và trả về danh sách Document."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").replace(" ", " ").replace("﻿", "")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": _title_of(content, fallback=path.stem),
                    "doc_type": doc_type,
                    "url": _url_of(content),
                },
            }
        )
    return documents


def _title_of(content: str, fallback: str) -> str:
    """Lấy tiêu đề từ dòng '# ...' đầu tiên do Task 3 ghi ra."""
    for line in content.split("\n", 20)[:20]:
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _url_of(content: str) -> str:
    """Lấy URL từ dòng '**Source:** ...' do Task 3 ghi ra.

    Chroma không lưu được giá trị None trong metadata, nên thiếu URL thì trả về
    chuỗi rỗng. Contract chấp nhận str hoặc None.
    """
    for line in content.split("\n", 20)[:20]:
        if line.startswith("**Source:**"):
            return line.replace("**Source:**", "").strip()
    return ""


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id ổn định và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        pieces = [piece.strip() for piece in splitter.split_text(document["content"])]
        index = 0
        for piece in pieces:
            # Chunk rỗng vi phạm contract và chỉ làm loãng BM25.
            if not piece:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": piece,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
            index += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk, giữ nguyên các field còn lại."""
    vectors = embed_texts([as_passage(chunk["content"]) for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB và ghi corpus dùng chung cho BM25."""
    collection = get_collection()
    batch = 256
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        collection.upsert(
            ids=[chunk["id"] for chunk in window],
            documents=[chunk["content"] for chunk in window],
            embeddings=[chunk["embedding"] for chunk in window],
            metadatas=[chunk["metadata"] for chunk in window],
        )

    CHUNK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_CACHE.write_text(
        json.dumps(
            [
                {"id": chunk["id"], "content": chunk["content"], "metadata": chunk["metadata"]}
                for chunk in chunks
            ],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    if not documents:
        raise SystemExit(f"Không tìm thấy Markdown nào trong {STANDARDIZED_DIR}. Chạy Task 3 trước.")

    chunks = chunk_documents(documents)
    print(f"Documents: {len(documents)} | Chunks: {len(chunks)}")
    print(f"Embedding: {EMBEDDING_MODEL} (provider={EMBEDDING_PROVIDER}, prefix={_NEEDS_PREFIX})")

    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks -> {CHROMA_DIR}")
    print(f"Corpus cache -> {CHUNK_CACHE}")


if __name__ == "__main__":
    run_pipeline()
