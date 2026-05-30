from __future__ import annotations

import numpy as np


class GoEncoder:
    """Compact 9x9 Go board encoder."""

    def encode(self, board: np.ndarray, turn: int = 1) -> np.ndarray:
        assert board.shape == (9, 9)
        liberties = self._compute_liberties(board)
        features: list[float] = []
        features.extend(board.flatten().tolist())
        features.extend((liberties / 4.0).flatten().tolist())
        black_stones = float((board == 1).sum())
        white_stones = float((board == -1).sum())
        empty_squares = float((board == 0).sum())
        features.extend(
            [
                black_stones / 81.0,
                white_stones / 81.0,
                empty_squares / 81.0,
                (black_stones - white_stones) / 81.0,
                float(turn),
                (black_stones + white_stones) / 81.0,
            ]
        )
        return np.asarray(features, dtype=np.float32)

    def _compute_liberties(self, board: np.ndarray) -> np.ndarray:
        liberties = np.zeros((9, 9), dtype=np.float32)
        for r in range(9):
            for c in range(9):
                if board[r, c] != 0:
                    count = 0
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 9 and 0 <= nc < 9 and board[nr, nc] == 0:
                            count += 1
                    liberties[r, c] = count
        return liberties
