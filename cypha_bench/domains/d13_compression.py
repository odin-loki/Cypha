"""Domain 13 — compression (Canterbury corpus or synthetic text)."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from bench_common import DEFAULT_SEED, data_path, finalize_domain, make_classifier, rng
from cypha_bench.encoders.text_encoder import CharNgramEncoder

CANTERBURY_FILES = {
    "alice29.txt": "literary prose",
    "asyoulik.txt": "Shakespeare play",
    "cp.html": "HTML markup",
    "fields.c": "C source code",
    "grammar.lsp": "Lisp code",
    "kennedy.xls": "binary spreadsheet",
    "lcet10.txt": "technical writing",
    "plrabn12.txt": "poetry",
    "ptt5": "binary fax",
    "sum": "binary executable",
    "xargs.1": "Unix man page",
}


def compute_compression_ratio(filepath: Path) -> float:
    original = filepath.read_bytes()
    compressed = gzip.compress(original, compresslevel=9)
    return len(original) / max(len(compressed), 1)


def _synthetic_corpus() -> dict[str, str]:
    g = rng(DEFAULT_SEED)
    prose = " ".join(["the quick brown fox jumps over the lazy dog"] * 200)
    code = "\n".join([f"int func_{i}() {{ return {i}; }}" for i in range(100)])
    binary = bytes(g.integers(0, 256, size=4096)).decode("latin-1")
    return {
        "synthetic_prose.txt": prose,
        "synthetic_code.c": code,
        "synthetic_binary.bin": binary,
    }


def _iter_files():
    base = data_path("canterbury")
    found = {}
    if base.exists():
        for name in CANTERBURY_FILES:
            p = base / name
            if p.exists():
                found[name] = p
    if found:
        return found

    synth_dir = data_path("canterbury_synthetic")
    synth_dir.mkdir(parents=True, exist_ok=True)
    for name, text in _synthetic_corpus().items():
        p = synth_dir / name
        p.write_text(text, encoding="utf-8", errors="ignore")
        found[name] = p
    return found


def _train_stream_on_text(text: str, max_steps: int = 2000):
    enc = CharNgramEncoder(n=5, vocab_size=128)
    enc.build_vocab(text[: min(len(text), 50000)])
    dim = max(enc.vocab_size, 1)
    clf = make_classifier(dim, seed=DEFAULT_SEED)
    losses, alphas = [], []
    n = len(text)
    for step in range(min(max_steps, n - enc.n)):
        window = text[step : step + enc.n + 20]
        x = enc.encode(window)
        if x.sum() == 0:
            continue
        next_char = text[step + enc.n] if step + enc.n < n else text[-1]
        label = f"c{ord(next_char) % 64}"
        loss = clf.train_step(x, label)
        losses.append(float(loss))
        diag = clf.diagnostics()
        alphas.append(float(diag.get("field_confidence", 0.5)))
    return {
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_alpha_proxy": float(np.mean(alphas)) if alphas else 0.5,
        "expert_count": int(clf.diagnostics().get("n_classes", 0)),
        "steps": len(losses),
    }


def experiment_13a_alpha_vs_compression():
    files = _iter_files()
    ratios, alphas, names = [], [], []
    for name, path in files.items():
        ratios.append(compute_compression_ratio(path))
        text = path.read_text(encoding="utf-8", errors="ignore")
        stats = _train_stream_on_text(text, max_steps=1500)
        alphas.append(stats["mean_alpha_proxy"])
        names.append(name)
    rho = float(spearmanr(alphas, ratios).correlation or 0.0) if len(ratios) > 2 else float("nan")
    return {
        "spearman_alpha_vs_gzip": rho,
        "files": names,
        "gzip_ratios": ratios,
        "mean_alpha_proxy": alphas,
    }


def experiment_13b_binary_vs_text():
    files = _iter_files()
    text_alphas, bin_alphas = [], []
    for name, path in files.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        stats = _train_stream_on_text(text, max_steps=800)
        if name.endswith((".xls", "ptt5", "sum", ".bin")) or "binary" in name:
            bin_alphas.append(stats["mean_alpha_proxy"])
        else:
            text_alphas.append(stats["mean_alpha_proxy"])
    return {
        "mean_alpha_text": float(np.mean(text_alphas)) if text_alphas else float("nan"),
        "mean_alpha_binary": float(np.mean(bin_alphas)) if bin_alphas else float("nan"),
    }


def run() -> dict:
    experiments = {
        "13A_alpha_vs_compression": experiment_13a_alpha_vs_compression(),
        "13B_binary_vs_text_alpha": experiment_13b_binary_vs_text(),
    }
    return finalize_domain("d13", experiments)


if __name__ == "__main__":
    print(run())
