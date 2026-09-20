"""UI layer for the RAG demo chatbot.

Deliberately free of any retrieval or generation logic: everything in this
package takes a ``trace``/``GenerationResult`` shaped like the frozen backend
contract and turns it into pixels. That separation is what lets the whole
interface be developed and demoed against ``ui.sample_trace`` while ``src/`` is
still moving.

Modules
-------
``theme``        colour tokens and the global stylesheet
``components``   header, source cards, badges, tiles, citation links
``charts``       Plotly figures (slope, scores, gauge, latency)
``pipeline_svg`` the animated hand-written pipeline diagram
``sample_trace`` realistic fixtures for DEMO_MODE
"""

from __future__ import annotations

__all__ = ["charts", "components", "pipeline_svg", "sample_trace", "theme"]
