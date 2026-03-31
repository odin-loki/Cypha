#!/usr/bin/env python3
"""One-shot: start cypha_rest with MKE sidecar, POST /update, check router loss vs mke_train_step fixture."""
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
FIX = ROOT / "parity_fixtures" / "mke_train_step"


def main() -> int:
    side_path = FIX / "sidecar.json"
    if not side_path.is_file():
        print("missing", side_path, file=sys.stderr)
        return 2
    side = json.loads(side_path.read_text(encoding="utf-8"))
    experts = {lbl: {"mu": 0.0, "var_ema": 0.25} for lbl in side["w_before"]}
    mke = {
        "d_in": side["d_in"],
        "D_rff": side["D_rff"],
        "temperature": side["temperature"],
        "forgetting_factor": side["forgetting_factor"],
        "pi_floor": 0.02,
        "rff_W_rowmajor": side["rff_W_rowmajor"],
        "rff_b": side["rff_b"],
        "w": side["w_before"],
        "P": side["P_before"],
        "gh_scales": side["gh_scales"],
    }
    reg = {"schema": 1, "experts": experts, "mke": mke}
    hp = {
        k: side[k]
        for k in (
            "world_lr",
            "delta_lr",
            "ood_sigma",
            "enc_lr",
            "replay_ratio",
            "replay_cap",
            "align_every",
            "temp_recalib_every",
        )
    }

    exe = os.environ.get("CYPHA_REST_BIN", str(ROOT / "native" / "build" / "cypha_rest"))
    if not os.path.isfile(exe):
        print("cypha_rest not found:", exe, file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        reg_path = os.path.join(td, "regression_head.json")
        hp_path = os.path.join(td, "train_hparams.json")
        Path(reg_path).write_text(json.dumps(reg), encoding="utf-8")
        Path(hp_path).write_text(json.dumps(hp), encoding="utf-8")

        port = 18765
        cmd = [
            exe,
            "--listen",
            f"127.0.0.1:{port}",
            "--cypha",
            str(FIX / "before.cypha"),
            "--f-field-json",
            str(FIX / "f_field.json"),
            "--regression-json",
            reg_path,
            "--train-hparams",
            hp_path,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(ROOT))
        try:
            last_err: Exception | None = None
            for _ in range(120):
                time.sleep(0.05)
                if proc.poll() is not None:
                    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                    print("server exited", proc.returncode, err[:800], file=sys.stderr)
                    return 1
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
                    break
                except (urllib.error.URLError, OSError) as e:
                    last_err = e
            else:
                print("timeout waiting for health", last_err, file=sys.stderr)
                return 1

            body = json.dumps(
                {
                    "input": side["x"],
                    "correct_label": side["router_train_label"],
                    "use_gh": True,
                    "regression_y": side["y"],
                    "router_train_label": side["router_train_label"],
                }
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/update",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            out = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            loss = float(out["loss"])
            exp_loss = float(side["expected_router_loss"])
            if abs(loss - exp_loss) > 1e-4:
                print("loss mismatch got", loss, "expected", exp_loss, file=sys.stderr)
                return 1
            print("smoke_mke_rest_update OK loss=", loss)
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
