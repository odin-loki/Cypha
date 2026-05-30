"""Domain 05 — chess position evaluation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchRegressor
from cypha_bench.common.baselines import offline_regression_baselines, sgd_online_regressor
from cypha_bench.common.metrics import cypha_metrics, evaluate_regression, online_train_regressor, save_figure, save_table, standardize_train_test
from cypha_bench.common.paths import DATA_DIR, scale
from cypha_bench.encoders import ChessEncoder


def _load_pgn_positions(pgn_path, max_games: int = 300, max_positions_per_game: int = 30):
    import chess
    import chess.pgn

    enc = ChessEncoder()
    x_rows, y_rows = [], []
    with open(pgn_path, encoding="utf-8", errors="replace") as handle:
        game_count = 0
        while game_count < max_games:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            elif result == "1/2-1/2":
                outcome = 0.0
            else:
                continue
            board = game.board()
            positions = 0
            for move in game.mainline_moves():
                if positions >= max_positions_per_game:
                    break
                board.push(move)
                x_rows.append(enc.encode(board))
                y_rows.append(outcome)
                positions += 1
            game_count += 1
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def _synthetic_chess(n_samples: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    enc = ChessEncoder()
    x_rows, y_rows = [], []
    for _ in range(n_samples):
        material = rng.integers(0, 12, size=12)
        features = enc.encode_synthetic(material)
        outcome = float(np.tanh(material[:6].sum() - material[6:].sum()))
        x_rows.append(features)
        y_rows.append(outcome)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def _load_data():
    pgn = DATA_DIR / "chess" / "kasparov.pgn"
    if pgn.exists():
        try:
            x, y = _load_pgn_positions(pgn, max_games=scale(300, 80))
            if len(x) > 100:
                return "kasparov_pgn", x, y
        except Exception:
            pass
    return "synthetic", *_synthetic_chess(scale(4000, 1000))


def run() -> dict:
    source, x, y = _load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = BenchRegressor(x_train.shape[1])
    online_train_regressor(model, x_train, y_train)
    scores = evaluate_regression(model, x_test, y_test)
    cypha = cypha_metrics(model, x_test, y_test, task="regression")
    baselines = offline_regression_baselines(x_train, y_train, x_test, y_test)
    sgd = sgd_online_regressor(x_train, y_train, x_test, y_test)

    metrics = {
        "domain": "d05_chess",
        "data_source": source,
        "n_samples": int(len(x)),
        "cypha_scores": scores,
        "cypha_metrics": cypha,
        "baselines": baselines,
        "sgd_online": sgd,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["CyphaDIF", "SGD online", "Ridge", "Random Forest"]
    rmse = [
        scores["rmse"],
        sgd["rmse"],
        baselines["ridge"]["rmse"],
        baselines["random_forest"]["rmse"],
    ]
    ax.bar(names, rmse, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Chess outcome regression ({source})")
    plt.tight_layout()

    save_table("d05_chess", metrics)
    save_figure(fig, "fig05_chess_rmse_vs_baselines")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run())
