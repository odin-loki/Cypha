"""Background Branch A route + CyphaLM or Ollama generation for chat."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..core.branch_a_router import BranchARouter, encode_prompt_chars
from ..core.ollama_client import ollama_generate


class BranchADispatchWorker(QThread):
    """
    Route text via Branch A, then stream CyphaLM (in-domain) or call Ollama (OOD).
    """

    status = Signal(str)
    route_done = Signal(dict)
    token_generated = Signal(dict)
    ollama_text = Signal(str)
    finished_generation = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        router: BranchARouter,
        text: str,
        lm_engine=None,
        *,
        epistemic_threshold: float | None = None,
        max_tokens: int = 120,
        temperature: float = 0.9,
        strategy: str = "top_p",
        top_p: float = 0.92,
    ) -> None:
        super().__init__()
        self._router = router
        self._text = text
        self._lm = lm_engine
        self._threshold = epistemic_threshold
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._strategy = strategy
        self._top_p = top_p

    def run(self) -> None:
        try:
            if not self._router.is_trained:
                self.status.emit("Training Branch A router (20 Newsgroups)…")
                self._router.train()

            self.status.emit("Routing query…")
            route = self._router.route(self._text, epistemic_threshold=self._threshold)
            self.route_done.emit(route)

            if route.get("abstain"):
                self.status.emit("OOD — calling Ollama fallback…")
                gen = ollama_generate(
                    self._text,
                    system=(
                        "You are a helpful assistant. The query was flagged as out-of-domain "
                        "for the Cypha router; answer directly and concisely."
                    ),
                )
                self.ollama_text.emit(str(gen.get("text", "")))
                self.finished_generation.emit()
                return

            if self._lm is None:
                self.finished_generation.emit()
                return

            self.status.emit("In-domain — CyphaLM generation…")
            vocab = int(getattr(self._lm.model.config, "vocab_size", 128))
            prompt_ids = encode_prompt_chars(self._text, vocab_size=vocab)
            for chunk in self._lm.stream_generate(
                prompt_ids,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                strategy=self._strategy,
                top_p=self._top_p,
            ):
                self.token_generated.emit(chunk)
                if chunk.get("done"):
                    break
            self.finished_generation.emit()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
