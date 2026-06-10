"""Run native parity/smoke tools from Windows (.exe) or via WSL (ELF in ``native/build*``)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def win_path_to_wsl(p: Path) -> str:
    r = p.resolve()
    s = str(r)
    if len(s) >= 3 and s[1] == ":" and s[2] in ("/", "\\"):
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    raise ValueError(f"Cannot map path to WSL: {s}")


def _looks_like_windows_abs(s: str) -> bool:
    return len(s) >= 3 and s[1] == ":" and s[0].isalpha() and s[2] in "/\\"


def _argv_for_wsl(str_argv: list[str]) -> list[str]:
    """Map Windows paths for ``wsl -e`` (including not-yet-created output files)."""
    out: list[str] = []
    for a in str_argv:
        if _looks_like_windows_abs(a):
            try:
                out.append(win_path_to_wsl(Path(a)))
            except ValueError:
                out.append(a)
        else:
            out.append(a)
    return out


def _windows_exe_candidates(stem: str) -> list[Path]:
    bases = [
        Path(r"C:\Temp\cypha_native_build6"),
        _ROOT / "native" / "build-mingw-w64",
        _ROOT / "native" / "build" / "Release",
        _ROOT / "native" / "build" / "Debug",
        _ROOT / "native" / "build",
    ]
    out: list[Path] = []
    for base in bases:
        out.append(base / f"{stem}.exe")
        out.append(base / "qt" / f"{stem}.exe")
    return out


def _elf_candidates(stem: str) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    # cypha_qt_* live under qt/; most tools at build root (native/qt/CMakeLists.txt).
    subdirs = ("", "qt")
    for b in ("build-wsl", "build-qt-stub", "build-ci-local", "build-exp", "build"):
        for sub in subdirs:
            p = _ROOT / "native" / b / sub / stem
            if p.is_file():
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
    native = _ROOT / "native"
    if native.is_dir():
        for build_dir in sorted(native.glob("build*")):
            if not build_dir.is_dir():
                continue
            for sub in subdirs:
                p = build_dir / sub / stem
                if p.is_file():
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        out.append(p)
    # Newest build wins when several native/build* trees exist (avoids a stale early tree
    # shadowing a freshly built cypha_qt_shell under e.g. build-qt-shell-test).
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _merged_env(extra: dict[str, str] | None) -> dict[str, str] | None:
    if not extra:
        return None
    env = os.environ.copy()
    env.update(extra)
    return env


def run_native_executable(
    stem: str,
    argv: list[str | Path],
    *,
    timeout: int = 60,
    env_override: str | None = None,
    extra_env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess | None:
    """
    Run ``native/.../{stem}`` or ``{stem}.exe``.

    ``env_override``: optional env var (e.g. ``CYPHA_EXPERIMENT_DB_SMOKE_BIN``) with a full path to the binary.
    ``extra_env``: merged into the subprocess environment (e.g. ``QT_QPA_PLATFORM`` for headless Qt).
    Returns ``None`` if no runnable binary was found.
    """
    str_argv = [str(x) for x in argv]
    env = _merged_env(extra_env)
    run_kw: dict = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "env": env,
    }
    if cwd is not None:
        run_kw["cwd"] = str(cwd)

    if env_override:
        raw = os.environ.get(env_override, "").strip()
        if raw:
            p = Path(raw)
            if p.is_file():
                return subprocess.run([str(p), *str_argv], **run_kw)

    if sys.platform == "win32":
        # Prefer WSL ELF (e.g. native/build-wsl) when present: matches Linux CTest and avoids
        # stale or misconfigured MinGW trees that may sit earlier on PATH.
        wsl = shutil.which("wsl")
        if wsl:
            for elf in _elf_candidates(stem):
                try:
                    cmd = [wsl, "-e", win_path_to_wsl(elf), *_argv_for_wsl(str_argv)]
                except ValueError:
                    continue
                return subprocess.run(cmd, timeout=timeout + 30, **{k: v for k, v in run_kw.items() if k != "timeout"})

        for exe in _windows_exe_candidates(stem):
            if exe.is_file():
                return subprocess.run([str(exe), *str_argv], **run_kw)

    else:
        for elf in _elf_candidates(stem):
            return subprocess.run([str(elf), *str_argv], **run_kw)

    return None
