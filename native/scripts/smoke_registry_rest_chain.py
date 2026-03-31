#!/usr/bin/env python3
"""E2E: empty registry → ``POST /register`` → ``POST /load`` → ``POST /predict`` (native ``cypha_rest``)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "parity_fixtures"


def main() -> int:
    if not (FIX / "reference.cypha").is_file() or not (FIX / "f_field.json").is_file():
        print("missing parity_fixtures", file=sys.stderr)
        return 2
    card = FIX / "registry_register" / "card.json"
    if not card.is_file():
        print("missing", card, file=sys.stderr)
        return 2

    exe = os.environ.get("CYPHA_REST_BIN", str(ROOT / "native" / "build" / "cypha_rest"))
    if not os.path.isfile(exe):
        print("cypha_rest not found:", exe, file=sys.stderr)
        return 2

    try:
        import numpy as np
    except ImportError:
        print("numpy required for x_input", file=sys.stderr)
        return 2

    reg_root = Path(tempfile.mkdtemp(prefix="cypha_reg_chain_"))
    port = 18766
    host = "127.0.0.1"
    cmd = [
        exe,
        "--listen",
        f"{host}:{port}",
        "--cypha",
        str(FIX / "reference.cypha"),
        "--f-field-json",
        str(FIX / "f_field.json"),
        "--registry",
        str(reg_root),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(ROOT))
    try:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                print("cypha_rest exited", proc.returncode, err[:600], file=sys.stderr)
                return 1
            try:
                urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.05)
        else:
            print("timeout waiting for health", file=sys.stderr)
            return 1

        name, version = "chain_smoke", "3.0.0"
        reg_body = json.dumps(
            {
                "name": name,
                "version": version,
                "model_cypha": str(FIX / "reference.cypha"),
                "card_json": str(card),
                "overwrite": True,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://{host}:{port}/register",
            data=reg_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r0 = urllib.request.urlopen(req, timeout=30)
        reg_out = json.loads(r0.read().decode())
        if not reg_out.get("registered"):
            print("register failed", reg_out, file=sys.stderr)
            return 1

        load_body = json.dumps({"name": name, "version": version}).encode()
        req2 = urllib.request.Request(
            f"http://{host}:{port}/load",
            data=load_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r1 = urllib.request.urlopen(req2, timeout=30)
        if r1.status != 200:
            print("load status", r1.status, file=sys.stderr)
            return 1
        loaded = json.loads(r1.read().decode())
        if "loaded" not in loaded:
            print("load body", loaded, file=sys.stderr)
            return 1

        exp = np.load(FIX / "expected.npz")
        x0 = exp["x_input"][0].astype(float).tolist()
        pred_body = json.dumps({"input": x0, "use_gh": True, "return_explanation": False}).encode()
        req3 = urllib.request.Request(
            f"http://{host}:{port}/predict",
            data=pred_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r2 = urllib.request.urlopen(req3, timeout=30)
        pred = json.loads(r2.read().decode())
        if "label" not in pred or "confidence" not in pred:
            print("predict", pred, file=sys.stderr)
            return 1

        print("smoke_registry_rest_chain OK label=", pred.get("label"), "conf=", pred.get("confidence"))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
