"""Background CyphaLM token streaming for the chat widget."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class LMGenerationWorker(QThread):
    """Runs CyphaLM.stream_generate in a worker thread."""

    token_generated = Signal(dict)
    finished_generation = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        lm_engine,
        prompt_ids: list[int],
        *,
        max_tokens: int = 120,
        temperature: float = 0.9,
        strategy: str = "top_p",
        top_k: int = 40,
        top_p: float = 0.92,
        uncertainty_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self._lm = lm_engine
        self._prompt_ids = prompt_ids
        self._kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "strategy": strategy,
            "top_k": top_k,
            "top_p": top_p,
            "uncertainty_threshold": uncertainty_threshold,
        }

    def run(self) -> None:
        try:
            for chunk in self._lm.stream_generate(self._prompt_ids, **self._kwargs):
                self.token_generated.emit(chunk)
                if chunk.get("done"):
                    break
            self.finished_generation.emit()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
