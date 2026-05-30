"""In-app HTML user guide (bundled ``help/guide.html``)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget


def _guide_html() -> str:
    p = Path(__file__).resolve().parent / "help" / "guide.html"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return "<p>Missing help/guide.html</p>"


class HelpWidget(QWidget):
    """Scrollable rich-text guide with internal anchor navigation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setSearchPaths([str(Path(__file__).resolve().parent / "help")])
        # QTextBrowser often ignores the app palette for document text — force contrast.
        self._browser.setStyleSheet(
            "QTextBrowser { background-color: #1e1e1e; color: #e6e6e6; }"
        )
        self._browser.document().setDefaultStyleSheet(
            "body { color: #e6e6e6; background-color: #1e1e1e; } "
            "h1 { color: #7eb8ff; } h2 { color: #9cdcfe; } "
            "a { color: #6cb6ff; } code { color: #e0c896; background: #2d2d2d; } "
            "p, li { color: #e0e0e0; }"
        )
        self._browser.setHtml(_guide_html())
        base = Path(__file__).resolve().parent / "help"
        if base.is_dir():
            self._browser.document().setBaseUrl(QUrl.fromLocalFile(str(base / "guide.html")))
        lay.addWidget(self._browser)

    def scroll_to_anchor(self, anchor: str) -> None:
        self._browser.scrollToAnchor(anchor)
