from __future__ import annotations

import numpy as np


class ChessEncoder:
    """Chess position encoder with python-chess when available."""

    PIECE_TYPES = ("pawn", "knight", "bishop", "rook", "queen", "king")

    def encode(self, board) -> np.ndarray:
        try:
            import chess
        except ImportError:
            return self.encode_synthetic(np.random.default_rng(0).integers(0, 12, size=12))

        if not isinstance(board, chess.Board):
            raise TypeError("Expected chess.Board when python-chess is installed.")

        features: list[float] = []
        for colour in (chess.WHITE, chess.BLACK):
            for piece_type in (
                chess.PAWN,
                chess.KNIGHT,
                chess.BISHOP,
                chess.ROOK,
                chess.QUEEN,
                chess.KING,
            ):
                features.append(float(len(board.pieces(piece_type, colour))))

        white_mob = {pt: 0 for pt in self.PIECE_TYPES}
        black_mob = {pt: 0 for pt in self.PIECE_TYPES}
        piece_map = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece is None:
                continue
            key = piece_map[piece.piece_type]
            if piece.color == chess.WHITE:
                white_mob[key] += 1
            else:
                black_mob[key] += 1
        for pt in self.PIECE_TYPES:
            features.append(float(white_mob[pt]))
            features.append(float(black_mob[pt]))

        centre = [chess.E4, chess.D4, chess.E5, chess.D5]
        for sq in centre:
            features.append(float(len(board.attackers(chess.WHITE, sq))))
            features.append(float(len(board.attackers(chess.BLACK, sq))))

        for colour in (chess.WHITE, chess.BLACK):
            king_sq = board.king(colour)
            if king_sq is not None:
                ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
                enemy = chess.BLACK if colour == chess.WHITE else chess.WHITE
                threats = sum(1 for sq in ring if board.is_attacked_by(enemy, sq))
                features.append(float(threats))
            else:
                features.append(0.0)

        for colour in (chess.WHITE, chess.BLACK):
            pawns = board.pieces(chess.PAWN, colour)
            files = [chess.square_file(sq) for sq in pawns]
            doubled = sum(1 for f in set(files) if files.count(f) > 1)
            features.append(float(doubled))

        for colour in (chess.WHITE, chess.BLACK):
            enemy = chess.BLACK if colour == chess.WHITE else chess.WHITE
            enemy_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, enemy)}
            passed = sum(
                1
                for sq in board.pieces(chess.PAWN, colour)
                if chess.square_file(sq) not in enemy_files
            )
            features.append(float(passed))

        for colour in (chess.WHITE, chess.BLACK):
            pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, colour)}
            isolated = sum(
                1
                for f in pawn_files
                if (f - 1) not in pawn_files and (f + 1) not in pawn_files
            )
            features.append(float(isolated))

        features.extend(
            [
                float(board.turn),
                float(board.has_kingside_castling_rights(chess.WHITE)),
                float(board.has_queenside_castling_rights(chess.WHITE)),
                float(board.has_kingside_castling_rights(chess.BLACK)),
                float(board.has_queenside_castling_rights(chess.BLACK)),
                float(board.ep_square is not None),
            ]
        )
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        total_material = sum(
            piece_values[pt] * len(board.pieces(pt, c))
            for pt in piece_values
            for c in (chess.WHITE, chess.BLACK)
        )
        features.append(total_material / 78.0)
        return np.asarray(features, dtype=np.float32)

    def encode_synthetic(self, material: np.ndarray) -> np.ndarray:
        vec = np.zeros(50, dtype=np.float32)
        vec[: min(len(material), 50)] = material[:50].astype(np.float32) / 10.0
        return vec
