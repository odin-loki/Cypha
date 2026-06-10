#!/usr/bin/env python3
"""One-shot: cypha_rest LM routes (load checkpoint → predict_next → generate → stream)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FIX = _ROOT / "parity_fixtures"
_CKPT = _FIX / "cyphalm_checkpoint" / "char_lstm" / "checkpoint.json"
_REF = _FIX / "reference.cypha"
_FF = _FIX / "f_field.json"


def _rest_bin() -> Path | None:
    env = os.environ.get("CYPHA_REST_BIN", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    for p in (
        _ROOT / "native" / "build-mingw-w64" / "cypha_rest.exe",
        Path(r"C:\Temp\cypha_native_build6\cypha_rest.exe"),
        _ROOT / "native" / "build" / "Release" / "cypha_rest.exe",
    ):
        if p.is_file():
            return p
    return None


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def main() -> int:
    exe = _rest_bin()
    if exe is None:
        print("skip: cypha_rest not built (set CYPHA_REST_BIN)", file=sys.stderr)
        return 0
    if not _CKPT.is_file() or not _REF.is_file() or not _FF.is_file():
        print("skip: parity_fixtures missing", file=sys.stderr)
        return 0

    try:
        import httpx
    except ImportError:
        print("skip: httpx not installed", file=sys.stderr)
        return 0

    port = _free_port()
    host = "127.0.0.1"
    cmd = [
        str(exe),
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(_REF),
        "--f-field-json",
        str(_FF),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(_ROOT))
    base = f"http://{host}:{port}"
    try:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                print(f"FAIL: cypha_rest exited ({proc.returncode}): {err[:400]}", file=sys.stderr)
                return 1
            try:
                if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            print("FAIL: server not ready", file=sys.stderr)
            return 1

        c = httpx.Client(base_url=base, timeout=30.0)
        assert c.get("/health").json().get("lm_loaded") is False

        r = c.post("/lm/load", json={"checkpoint_path": str(_CKPT)})
        if r.status_code != 200:
            print(f"FAIL /lm/load: {r.status_code} {r.text}", file=sys.stderr)
            return 1
        assert c.get("/health").json().get("lm_loaded") is True

        ev = _CKPT.parent / "sidecar.json"
        side = json.loads(ev.read_text(encoding="utf-8"))
        tid = int(side["eval_ids"][0])
        pn = c.post("/lm/predict_next", json={"token_id": tid})
        if pn.status_code != 200:
            print(f"FAIL /lm/predict_next: {pn.text}", file=sys.stderr)
            return 1
        assert "log_probs" in pn.json()

        gen = c.post(
            "/generate",
            json={"prompt_ids": [tid], "max_tokens": 4, "strategy": "greedy"},
        )
        if gen.status_code != 200:
            print(f"FAIL /generate: {gen.text}", file=sys.stderr)
            return 1
        gbody = gen.json()
        assert len(gbody.get("generated_ids", [])) == 4

        with c.stream(
            "POST",
            "/generate/stream",
            json={"prompt_ids": [tid], "max_tokens": 3, "strategy": "greedy"},
        ) as resp:
            if resp.status_code != 200:
                print(f"FAIL /generate/stream: {resp.status_code}", file=sys.stderr)
                return 1
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    chunks.append(json.loads(line[6:]))
            assert any(ch.get("done") for ch in chunks)

        print("OK cyphalm_rest_lm smoke")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
