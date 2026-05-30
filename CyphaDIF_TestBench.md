# CyphaDIF Comprehensive Test Bench
**For:** Compose 2.5 in Cursor  
**Author:** Odin Loch  
**Purpose:** Full empirical characterisation of CyphaDIF across every domain it can operate in. Produces a consolidated report with figures for all domains.

---

## How to Read This Document

This document is a complete, self-contained implementation specification. Read it top to bottom before writing any code. Every section contains:
- What data to use and exactly how to acquire it (single console command)
- How to encode non-tabular data into feature vectors CyphaDIF can consume
- The exact experiment protocol
- Metrics to record and baselines to beat
- CyphaDIF-specific measurements that no other model produces
- The figure this experiment contributes to the final report

Do not skip the **CyphaDIF API Reference** section (Section 3). Every encoding and training loop in this document assumes you have read it.

All code targets **Python 3.11+**, **NumPy**, **SciPy**, **scikit-learn**, and CyphaDIF from Odin's private repository (GitHub: odin-loki). No PyTorch. No TensorFlow. No Keras. External libraries listed in Section 2 are all installable via `pip`.

---

## 1. Repository Structure

```
cypha_bench/
├── setup/
│   ├── acquire_data.sh            # All data downloads in one script
│   └── verify_data.py             # Confirms all datasets are present and valid
├── data/                          # All datasets land here
│   ├── mnist/
│   ├── wikitext2/
│   ├── 20news/                    # Auto-created by sklearn
│   ├── nsl_kdd/
│   ├── ecg5000/
│   ├── canterbury/
│   ├── gutenberg/
│   ├── chess/
│   ├── financial/
│   └── uci/
├── encoders/
│   ├── __init__.py
│   ├── image_encoder.py           # MNIST pixel + HOG + patch strategies
│   ├── text_encoder.py            # TF-IDF, char n-gram, token window
│   ├── chess_encoder.py           # Board position → feature vector
│   ├── go_encoder.py              # 9×9 board → feature vector
│   ├── poker_encoder.py           # Hand state → feature vector
│   ├── timeseries_encoder.py      # Sliding window → statistical features
│   ├── document_encoder.py        # TF-IDF doc vectors
│   └── audio_encoder.py           # MFCC/FFT for any audio signal
├── domains/
│   ├── d01_statistical_baselines.py
│   ├── d02_regression.py
│   ├── d03_classification.py
│   ├── d04_generation_language.py
│   ├── d05_chess.py
│   ├── d06_go.py
│   ├── d07_poker.py
│   ├── d08_computer_vision.py
│   ├── d09_documents.py
│   ├── d10_time_series.py
│   ├── d11_reinforcement_learning.py
│   ├── d12_anomaly_detection.py
│   ├── d13_compression.py
│   ├── d14_symbolic_regression.py
│   ├── d15_adversarial_robustness.py
│   ├── d16_multitask.py
│   └── d17_cyphalm_integration.py
├── cross_domain/
│   ├── uncertainty_calibration.py  # ECE across all domains
│   ├── online_adaptation.py        # Adaptation speed across all domains
│   ├── forgetting_resistance.py    # A→B→A protocol across domain pairs
│   └── alpha_spectrum_global.py    # Grand Unified Law α across all domains
├── report/
│   ├── figures/                    # All experiment figures land here
│   ├── tables/                     # All experiment tables land here
│   └── generate_report.py          # Assembles everything into a PDF report
├── run_all.py                      # Master runner: all 17 domains in sequence
└── requirements.txt
```

---

## 2. Environment Setup

### 2.1 Python Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib pandas tqdm galois \
            python-chess sgfmill treys yfinance ucimlrepo \
            pytest pytest-benchmark reportlab pillow
```

- `galois` — GF(2ⁿ) arithmetic for Izaac embeddings
- `python-chess` — Chess board parsing and move generation
- `sgfmill` — Go SGF file parsing and board state
- `treys` — Poker hand evaluator (Cactus Kev algorithm)
- `yfinance` — Yahoo Finance price data (no API key needed)
- `ucimlrepo` — Programmatic access to UCI ML repository datasets
- `reportlab` — PDF report generation
- `pillow` — Image loading and preprocessing

### 2.2 Data Acquisition

Create `setup/acquire_data.sh` with the following content and run it once:

```bash
#!/bin/bash
# Run from the cypha_bench/ root directory
# WSL: bash setup/acquire_data.sh
# Windows: run each wget line manually in PowerShell with Invoke-WebRequest if wget not available

mkdir -p data/mnist data/wikitext2 data/nsl_kdd data/ecg5000 \
         data/canterbury data/gutenberg data/chess data/financial data/uci

# --- MNIST (11MB total, 4 files) ---
cd data/mnist
wget -nc http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -nc http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -nc http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -nc http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip -k *.gz
cd ../..

# --- WikiText-2 (~4MB compressed) ---
cd data/wikitext2
wget -nc https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -n wikitext-2-v1.zip
cd ../..

# --- NSL-KDD anomaly detection (~7MB) ---
cd data/nsl_kdd
wget -nc https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget -nc https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
cd ../..

# --- ECG5000 time series (~3MB) ---
cd data/ecg5000
wget -nc http://www.timeseriesclassification.com/Downloads/ECG5000.zip
unzip -n ECG5000.zip
cd ../..

# --- Canterbury Corpus compression benchmark (~2MB) ---
cd data/canterbury
wget -nc http://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz
tar -xzf cantrbry.tar.gz
cd ../..

# --- Project Gutenberg texts (3 books, ~2MB total) ---
cd data/gutenberg
wget -nc -O moby_dick.txt      https://www.gutenberg.org/files/2701/2701-0.txt
wget -nc -O sherlock_holmes.txt https://www.gutenberg.org/files/1661/1661-0.txt
wget -nc -O alice.txt           https://www.gutenberg.org/files/11/11-0.txt
cd ../..

# --- Chess PGN — Kasparov complete games (~2MB) ---
cd data/chess
wget -nc -O kasparov.pgn https://www.pgnmentor.com/players/Kasparov.zip
unzip -n Kasparov.zip
cd ../..

echo "Data acquisition complete."
echo "20 Newsgroups will be auto-downloaded by sklearn on first run (~14MB)."
echo "Financial data will be pulled live by yfinance on first run (tiny)."
```

### 2.3 Verify Data

```python
# setup/verify_data.py
import os, sys

required = [
    "data/mnist/train-images-idx3-ubyte",
    "data/wikitext2/wikitext-2/wiki.train.tokens",
    "data/nsl_kdd/KDDTrain+.txt",
    "data/nsl_kdd/KDDTest+.txt",
    "data/ecg5000/ECG5000_TRAIN.txt",
    "data/ecg5000/ECG5000_TEST.txt",
    "data/canterbury/alice29.txt",
    "data/gutenberg/moby_dick.txt",
    "data/gutenberg/sherlock_holmes.txt",
]

missing = [p for p in required if not os.path.exists(p)]
if missing:
    print("MISSING FILES:")
    for p in missing: print(f"  {p}")
    sys.exit(1)
else:
    print("All required data files present.")
```

---

## 3. CyphaDIF API Reference

**Read this section before implementing any domain.** All experiments import from CyphaDIF. The API below is the complete surface used across this bench. Check the actual import paths against `odin-loki` GitHub before running — the interfaces below reflect the documented design.

```python
# Core imports (verify exact paths against repo)
from cyphadif import CyphaDIF
from cyphadif.regression import DIFRegressor
from cyphadif.classification import DIFClassifier
from cyphadif.encoders import VectorEncoder, WindowEncoder

# Instantiation
dif = CyphaDIF(
    encoder=VectorEncoder(input_dim),  # Wraps a numpy array input
    field_dim=160,                      # DIF field dimension per expert
    n_experts=0,                        # 0 = dynamic growth
)

# One online training step
loss = dif.train_step(x, label)        # x: np.ndarray, label: str or int

# Prediction
result = dif.predict(x)
# result contains:
#   result['probs']           : np.ndarray — class routing probabilities
#   result['prediction']      : str/int    — argmax prediction
#   result['epistemic_var']   : float      — parameter uncertainty (lossless residual proxy)
#   result['aleatoric_var']   : float      — irreducible noise
#   result['active_experts']  : int        — experts with p(k|x) > 0.01
#   result['expert_alphas']   : np.ndarray — GRIA α per active expert

# Regression wrapper
reg = DIFRegressor(encoder=VectorEncoder(input_dim), field_dim=160)
reg.train_step(x, y_float)
y_pred, epistemic_var, aleatoric_var = reg.predict(x)

# Classification wrapper
clf = DIFClassifier(encoder=VectorEncoder(input_dim), field_dim=160)
clf.train_step(x, class_label)
pred_class, probs, epistemic_var = clf.predict(x)

# Expert inspection
n_experts   = dif.expert_count()
alpha_vec   = dif.alpha_per_expert()          # np.ndarray (K,) — Grand Unified Law α
mean_alpha  = dif.mean_alpha()
expert_means = dif.expert_means()             # dict: expert_id → mean feature vector

# Serialisation
dif.save("model.cypha")
dif.load("model.cypha")

# Reset
dif.reset()
```

### 3.1 Standard Metrics Helper

Every domain computes the same CyphaDIF-specific metrics. Use this helper:

```python
# In every domain script, import and call this after training
def cypha_metrics(model, X_test, y_test, task='classification'):
    """
    Computes CyphaDIF-specific measurements that no baseline can produce.
    Returns a dict ready for the report.
    """
    epistemic_vars, aleatoric_vars, predictions, losses = [], [], [], []

    for x, y in zip(X_test, y_test):
        if task == 'classification':
            pred, probs, ep_var = model.predict(x) if hasattr(model, 'predict') else (None, None, None)
            al_var = 0.0  # Update if available from model
        else:
            pred, ep_var, al_var = model.predict(x)

        epistemic_vars.append(ep_var)
        aleatoric_vars.append(al_var)
        predictions.append(pred)

    ep = np.array(epistemic_vars)
    al = np.array(aleatoric_vars)

    return {
        'mean_epistemic_var':    ep.mean(),
        'std_epistemic_var':     ep.std(),
        'mean_aleatoric_var':    al.mean(),
        'expert_count':          model.expert_count(),
        'mean_alpha':            model.mean_alpha(),
        'alpha_distribution':    model.alpha_per_expert(),
        'fraction_edge_of_chaos': np.mean(np.abs(model.alpha_per_expert() - 0.5) < 0.1),
        'uncertainty_rank_correlation': _uncertainty_vs_error_correlation(
            np.array(predictions), np.array(y_test), ep
        ),
    }

def _uncertainty_vs_error_correlation(predictions, targets, epistemic_vars):
    """Spearman rank correlation between epistemic_var and per-sample error."""
    from scipy.stats import spearmanr
    errors = np.abs(predictions.astype(float) - targets.astype(float))
    rho, _ = spearmanr(epistemic_vars, errors)
    return rho
```

### 3.2 Baseline Suite

Every domain runs the same set of baselines for comparison:

| Baseline | sklearn class | Notes |
|---|---|---|
| Majority class / mean | DummyClassifier / DummyRegressor | Lower bound |
| k-NN (k=5) | KNeighborsClassifier/Regressor | Non-parametric reference |
| Random Forest (100 trees) | RandomForestClassifier/Regressor | Strong offline baseline |
| Gradient Boosting | GradientBoostingClassifier/Regressor | Strong offline baseline |
| Logistic Regression / Ridge | LogisticRegression / Ridge | Linear baseline |
| SGD Classifier/Regressor | SGDClassifier/Regressor | Online learning baseline |

CyphaDIF must be compared against SGD as the primary online learning baseline (same training regime — one sample at a time, no batch accumulation).

---

## 4. Encoding Strategies

Non-tabular data must be converted to fixed-length numpy feature vectors before CyphaDIF can process it. This section defines the encoding for each modality.

### 4.1 Image Encoding (MNIST)

```python
# encoders/image_encoder.py
import numpy as np
from scipy.ndimage import uniform_filter

class ImageEncoder:
    """Three encoding strategies for 28x28 greyscale images."""

    def raw_pixels(self, img: np.ndarray) -> np.ndarray:
        """Flatten and normalise. Output dim: 784."""
        return img.flatten().astype(np.float32) / 255.0

    def hog_features(self, img: np.ndarray, cell_size=4, n_bins=9) -> np.ndarray:
        """
        Histogram of Oriented Gradients. Output dim: ~144 (depends on cell_size).
        Pure numpy/scipy — no sklearn dependency.
        """
        img = img.astype(np.float32) / 255.0
        gx = np.gradient(img, axis=1)
        gy = np.gradient(img, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) % np.pi  # 0 to pi

        n_cells_x = img.shape[1] // cell_size
        n_cells_y = img.shape[0] // cell_size
        hog = np.zeros((n_cells_y, n_cells_x, n_bins))

        for b in range(n_bins):
            angle_low  = b * np.pi / n_bins
            angle_high = (b + 1) * np.pi / n_bins
            mask = (angle >= angle_low) & (angle < angle_high)
            weighted = magnitude * mask
            for cy in range(n_cells_y):
                for cx in range(n_cells_x):
                    cell = weighted[cy*cell_size:(cy+1)*cell_size,
                                    cx*cell_size:(cx+1)*cell_size]
                    hog[cy, cx, b] = cell.sum()

        return hog.flatten()

    def patch_features(self, img: np.ndarray, patch_size=7, stride=7) -> list[np.ndarray]:
        """
        Splits image into non-overlapping patches.
        Returns a list of patch feature vectors (each is raw_pixels of the patch).
        Use for patch-routing experiment where each patch is routed independently.
        """
        patches = []
        img_norm = img.astype(np.float32) / 255.0
        h, w = img.shape
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img_norm[y:y+patch_size, x:x+patch_size]
                patches.append(patch.flatten())
        return patches
```

### 4.2 Chess Position Encoding

```python
# encoders/chess_encoder.py
import chess
import numpy as np

class ChessEncoder:
    """
    Encodes a chess board position as a 113-dimensional feature vector.
    No Stockfish required — purely structural features.
    """

    PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]

    def encode(self, board: chess.Board) -> np.ndarray:
        features = []

        # 1. Material count per piece type per colour (12 features)
        for colour in [chess.WHITE, chess.BLACK]:
            for pt in self.PIECE_TYPES:
                features.append(len(board.pieces(pt, colour)))

        # 2. Piece mobility — number of legal moves per piece type (12 features)
        # Push each legal move and count
        white_mobility = {pt: 0 for pt in self.PIECE_TYPES}
        black_mobility = {pt: 0 for pt in self.PIECE_TYPES}
        for move in board.legal_moves:
            from_sq = move.from_square
            piece = board.piece_at(from_sq)
            if piece:
                if piece.color == chess.WHITE:
                    white_mobility[piece.piece_type] += 1
                else:
                    black_mobility[piece.piece_type] += 1
        for pt in self.PIECE_TYPES:
            features.append(white_mobility[pt])
            features.append(black_mobility[pt])

        # 3. Centre control — attacks on e4,d4,e5,d5 (4 features each colour = 8)
        centre_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for sq in centre_squares:
            features.append(len(board.attackers(chess.WHITE, sq)))
            features.append(len(board.attackers(chess.BLACK, sq)))

        # 4. King safety — number of attackers near king (2 features)
        for colour in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(colour)
            if king_sq is not None:
                ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
                enemy = chess.BLACK if colour == chess.WHITE else chess.WHITE
                threats = sum(1 for sq in ring if board.is_attacked_by(enemy, sq))
                features.append(threats)
            else:
                features.append(0)

        # 5. Pawn structure (6 features)
        # Doubled pawns per colour
        for colour in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, colour)
            files = [chess.square_file(sq) for sq in pawns]
            doubled = sum(1 for f in set(files) if files.count(f) > 1)
            features.append(doubled)

        # Passed pawns per colour (simplified: no enemy pawn on same file ahead)
        for colour in [chess.WHITE, chess.BLACK]:
            enemy = chess.BLACK if colour == chess.WHITE else chess.WHITE
            enemy_pawns = board.pieces(chess.PAWN, enemy)
            enemy_files = set(chess.square_file(sq) for sq in enemy_pawns)
            passed = sum(1 for sq in board.pieces(chess.PAWN, colour)
                         if chess.square_file(sq) not in enemy_files)
            features.append(passed)

        # Isolated pawns per colour (no friendly pawns on adjacent files)
        for colour in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, colour)
            pawn_files = set(chess.square_file(sq) for sq in pawns)
            isolated = sum(1 for f in pawn_files
                           if (f-1) not in pawn_files and (f+1) not in pawn_files)
            features.append(isolated)

        # 6. Turn, castling rights, en passant available (5 features)
        features.append(int(board.turn))
        features.append(int(board.has_kingside_castling_rights(chess.WHITE)))
        features.append(int(board.has_queenside_castling_rights(chess.WHITE)))
        features.append(int(board.has_kingside_castling_rights(chess.BLACK)))
        features.append(int(board.has_queenside_castling_rights(chess.BLACK)))
        features.append(int(board.ep_square is not None))

        # 7. Game phase estimate: total material value / 78 (1 feature)
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        total_material = sum(
            piece_values[pt] * len(board.pieces(pt, c))
            for pt in piece_values for c in [chess.WHITE, chess.BLACK]
        )
        features.append(total_material / 78.0)  # Normalised

        return np.array(features, dtype=np.float32)  # 113-dimensional
```

### 4.3 Go Position Encoding (9×9)

```python
# encoders/go_encoder.py
import numpy as np

class GoEncoder:
    """
    Encodes a 9x9 Go board position as a feature vector.
    Input: board as 9x9 numpy array where 1=Black, -1=White, 0=Empty.
    Output: 108-dimensional feature vector.
    """

    def encode(self, board: np.ndarray, turn: int = 1) -> np.ndarray:
        assert board.shape == (9, 9)
        features = []

        # 1. Raw board state (81 features): -1, 0, +1
        features.extend(board.flatten().tolist())

        # 2. Liberty counts per stone (81 features): number of liberties for stone at each position
        liberties = self._compute_liberties(board)
        features.extend(liberties.flatten().tolist())

        # Normalise liberty counts
        features[-81:] = [x / 4.0 for x in features[-81:]]  # max 4 liberties per stone

        # 3. Territory influence map (simplified: sum of stone values in 3x3 neighbourhood)
        influence = np.zeros((9, 9))
        for r in range(9):
            for c in range(9):
                nbr = board[max(0,r-1):r+2, max(0,c-1):c+2]
                influence[r, c] = nbr.sum() / 9.0
        features.extend(influence.flatten().tolist())

        # Wait - that's 81+81+81 = 243, let's trim to just first two + summary stats

        # Reset and use a compact encoding
        features = []
        features.extend(board.flatten().tolist())           # 81
        features.extend(liberties.flatten().tolist())        # 81 (normalised below)

        # 4. Summary stats (6 features)
        black_stones = (board == 1).sum()
        white_stones = (board == -1).sum()
        empty_squares = (board == 0).sum()
        features.append(black_stones / 81.0)
        features.append(white_stones / 81.0)
        features.append(empty_squares / 81.0)
        features.append((black_stones - white_stones) / 81.0)  # territory diff
        features.append(turn)                                    # whose turn
        features.append(float(black_stones + white_stones) / 81.0)  # game progress

        return np.array(features, dtype=np.float32)  # 168-dimensional

    def _compute_liberties(self, board: np.ndarray) -> np.ndarray:
        liberties = np.zeros((9, 9))
        for r in range(9):
            for c in range(9):
                if board[r, c] != 0:
                    count = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < 9 and 0 <= nc < 9 and board[nr, nc] == 0:
                            count += 1
                    liberties[r, c] = count
        return liberties
```

### 4.4 Poker Hand Encoding

```python
# encoders/poker_encoder.py
import numpy as np
from treys import Deck, Evaluator, Card

class PokerEncoder:
    """
    Encodes a Texas Hold'em situation as a feature vector.
    Uses the treys library for hand strength evaluation.
    """

    def __init__(self):
        self.evaluator = Evaluator()

    def encode_hand_situation(self, hole_cards: list, community_cards: list,
                               pot_size: float, stack_size: float,
                               position: int) -> np.ndarray:
        """
        hole_cards: list of 2 treys Card objects
        community_cards: list of 0, 3, 4, or 5 treys Card objects
        pot_size: float, normalised by starting stack
        stack_size: float, normalised by starting stack
        position: 0=early, 1=middle, 2=late/button

        Returns 20-dimensional feature vector.
        """
        features = []

        # 1. Hand strength (1 feature): normalised rank 0-1 (0=best, 1=worst)
        if len(community_cards) >= 3:
            rank = self.evaluator.evaluate(community_cards, hole_cards)
            hand_strength = 1.0 - (rank / 7462.0)
        else:
            # Pre-flop: estimate from hole card ranks and suitedness
            hand_strength = self._preflop_strength(hole_cards)
        features.append(hand_strength)

        # 2. Hand class (9 binary features): one-hot of hand category
        if len(community_cards) >= 3:
            hand_class = self.evaluator.get_rank_class(
                self.evaluator.evaluate(community_cards, hole_cards)
            )
            hand_class_idx = hand_class - 1  # 1-9 → 0-8
        else:
            hand_class_idx = 8  # Unknown pre-flop
        one_hot = [0.0] * 9
        one_hot[hand_class_idx] = 1.0
        features.extend(one_hot)

        # 3. Hole card ranks (2 features, normalised 2-14 → 0-1)
        for card in hole_cards[:2]:
            features.append((Card.get_rank_int(card) - 2) / 12.0)

        # 4. Suitedness (1 feature)
        suited = int(Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1]))
        features.append(float(suited))

        # 5. Street (1 feature: 0=preflop, 0.33=flop, 0.67=turn, 1=river)
        street = len(community_cards) / 5.0
        features.append(street)

        # 6. Pot odds and stack (3 features)
        features.append(min(pot_size, 10.0) / 10.0)
        features.append(min(stack_size, 10.0) / 10.0)
        features.append(float(position) / 2.0)

        # 7. Outs estimate (1 feature, approximated)
        outs = self._estimate_outs(hole_cards, community_cards)
        features.append(outs / 47.0)

        assert len(features) == 18
        return np.array(features, dtype=np.float32)

    def _preflop_strength(self, hole_cards):
        """Chen formula approximation, normalised."""
        r1 = Card.get_rank_int(hole_cards[0])
        r2 = Card.get_rank_int(hole_cards[1])
        high = max(r1, r2) / 12.0
        gap = abs(r1 - r2) / 12.0
        suited = float(Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1]))
        return max(0.0, min(1.0, high - 0.3 * gap + 0.1 * suited))

    def _estimate_outs(self, hole_cards, community_cards):
        """Rough outs estimate based on draws present."""
        if len(community_cards) < 3:
            return 0
        # Very rough: if suited hole cards and 2+ of that suit on board = flush draw
        h_suits = [Card.get_suit_int(c) for c in hole_cards]
        c_suits = [Card.get_suit_int(c) for c in community_cards]
        for s in set(h_suits):
            if h_suits.count(s) == 2 and c_suits.count(s) >= 2:
                return 9  # Flush draw
        return 2  # Default: backdoor or nothing useful

    def generate_random_situation(self, rng: np.random.Generator) -> tuple:
        """Generate a random valid poker hand situation for synthetic training."""
        deck = Deck()
        deck.shuffle()
        hole  = deck.draw(2)
        flop  = deck.draw(3)
        community = flop if rng.random() > 0.5 else flop + deck.draw(1)
        pot   = float(rng.integers(1, 50))
        stack = float(rng.integers(10, 200))
        pos   = int(rng.integers(0, 3))
        vec   = self.encode_hand_situation(hole, community, pot/100, stack/200, pos)
        # Target: 1 if hand_strength > 0.5 else 0 (should fold/call/raise)
        hs    = vec[0]
        label = 'raise' if hs > 0.65 else ('call' if hs > 0.35 else 'fold')
        return vec, label
```

### 4.5 Time Series Encoding

```python
# encoders/timeseries_encoder.py
import numpy as np
from scipy.fft import rfft

class TimeSeriesEncoder:
    """
    Sliding window encoder for time series data.
    Extracts statistical + frequency features from a window of samples.
    """

    def __init__(self, window_size=50, n_fft_coeffs=10):
        self.window_size = window_size
        self.n_fft_coeffs = n_fft_coeffs
        # Feature count: 8 stat features + n_fft_coeffs = 18 by default
        self.feature_dim = 8 + n_fft_coeffs

    def encode_window(self, window: np.ndarray) -> np.ndarray:
        """
        Input: 1D array of length window_size.
        Output: feature_dim-dimensional float vector.
        """
        assert len(window) == self.window_size
        features = []

        # Statistical moments (8 features)
        features.append(float(np.mean(window)))
        features.append(float(np.std(window)))
        features.append(float(np.min(window)))
        features.append(float(np.max(window)))
        features.append(float(np.percentile(window, 25)))
        features.append(float(np.percentile(window, 75)))
        features.append(float(np.mean(np.abs(np.diff(window)))))   # mean absolute change
        features.append(float(np.sum(np.sign(np.diff(window)) != 0)))  # zero crossings

        # FFT coefficients (magnitude of first n_fft_coeffs)
        fft_coeffs = np.abs(rfft(window))
        features.extend(fft_coeffs[:self.n_fft_coeffs].tolist())

        return np.array(features, dtype=np.float32)

    def sliding_windows(self, series: np.ndarray, step=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, indices) where X[i] is the encoded window ending at series[indices[i]].
        """
        n = len(series)
        indices = list(range(self.window_size, n, step))
        X = np.array([
            self.encode_window(series[i - self.window_size:i])
            for i in indices
        ])
        return X, np.array(indices)
```

### 4.6 Text / Document Encoding

```python
# encoders/text_encoder.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import issparse

class TextEncoder:
    """TF-IDF and character n-gram encoders for text documents."""

    def __init__(self, max_features=1000, analyzer='word', ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer=analyzer,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.fitted = False

    def fit(self, documents: list[str]):
        self.vectorizer.fit(documents)
        self.fitted = True

    def encode(self, text: str) -> np.ndarray:
        assert self.fitted, "Call fit() first."
        vec = self.vectorizer.transform([text])
        if issparse(vec):
            vec = vec.toarray()
        return vec.flatten().astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        assert self.fitted
        vecs = self.vectorizer.transform(texts)
        if issparse(vecs):
            vecs = vecs.toarray()
        return vecs.astype(np.float32)

class CharNgramEncoder:
    """Character n-gram window encoder for streaming text (used in language modelling)."""

    def __init__(self, n=5, vocab_size=200):
        self.n = n
        self.vocab_size = vocab_size
        self._vocab = {}

    def build_vocab(self, text: str):
        from collections import Counter
        ngrams = [text[i:i+self.n] for i in range(len(text)-self.n)]
        counts = Counter(ngrams)
        self._vocab = {ng: i for i, (ng, _) in
                       enumerate(counts.most_common(self.vocab_size))}

    def encode(self, window: str) -> np.ndarray:
        """Encodes the last n characters as a one-hot-style bag of n-grams."""
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for i in range(len(window) - self.n):
            ng = window[i:i+self.n]
            if ng in self._vocab:
                vec[self._vocab[ng]] += 1.0
        norm = vec.sum()
        if norm > 0:
            vec /= norm
        return vec
```

---

## 5. Domain 01 — Statistical Baselines and Sanity Checks

**Purpose:** Verify CyphaDIF behaves correctly on trivially structured data before running any real experiments. These tests establish lower and upper bounds.

**Data:** Synthetic — generated in-script, no download.

**Protocol:**

```python
# domains/d01_statistical_baselines.py

import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs

rng = np.random.default_rng(42)

TASKS = [
    # (name, generator, task_type, expected_accuracy_lower_bound)
    ("linearly_separable_2class",
     lambda: make_classification(n_samples=2000, n_features=10, n_informative=5,
                                  n_redundant=2, random_state=42),
     "classification", 0.90),

    ("4_gaussian_blobs",
     lambda: make_blobs(n_samples=2000, n_features=8, centers=4,
                        cluster_std=1.5, random_state=42),
     "classification", 0.85),

    ("high_dim_noisy",
     lambda: make_classification(n_samples=2000, n_features=100, n_informative=10,
                                  n_redundant=40, random_state=42),
     "classification", 0.75),

    ("linear_regression",
     lambda: make_regression(n_samples=2000, n_features=20, noise=0.1, random_state=42),
     "regression", None),

    ("nonlinear_regression_sinusoidal",
     lambda: _make_sinusoidal(2000, rng),
     "regression", None),

    ("single_concept_drift",
     lambda: _make_drift(2000, rng),
     "classification", 0.80),  # Accuracy after drift, testing online adaptation

    ("pure_noise",
     lambda: (rng.standard_normal((1000, 10)), rng.integers(0, 2, 1000)),
     "classification", 0.52),  # Should stay near 50% — tests that CyphaDIF doesn't overfit noise

    ("identical_inputs_different_labels",
     lambda: _make_contradictory(500, rng),
     "classification", 0.50),  # Irreducible: aleatoric_var should be high
]

def _make_sinusoidal(n, rng):
    X = rng.standard_normal((n, 1))
    y = np.sin(3 * X[:, 0]) + 0.2 * rng.standard_normal(n)
    return X, y

def _make_drift(n, rng):
    X1 = rng.standard_normal((n//2, 10))
    y1 = (X1[:, 0] > 0).astype(int)
    X2 = rng.standard_normal((n//2, 10))
    y2 = (X2[:, 1] > 0).astype(int)  # Different decision boundary after drift
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    return X, y

def _make_contradictory(n, rng):
    """Same x maps to different y — tests aleatoric uncertainty."""
    X = np.tile(rng.standard_normal((n//2, 10)), (2, 1))
    y = np.concatenate([np.zeros(n//2), np.ones(n//2)]).astype(int)
    return X, y
```

**Assertions (all must pass before proceeding to other domains):**

1. On `pure_noise`, `mean_epistemic_var` should be HIGHER than on `linearly_separable_2class` after same number of training steps — model is less confident on noise than on structure.
2. On `identical_inputs_different_labels`, `mean_aleatoric_var` should be at least 2× higher than `mean_epistemic_var` — this is irreducible noise, not parameter uncertainty.
3. On `single_concept_drift`, epistemic_var should spike at the drift point (step 1000) and then decrease again as the model adapts — demonstrating online learning in action.
4. Expert count on `4_gaussian_blobs` should converge to 4 ± 2 after sufficient training.
5. Expert count on `pure_noise` should either stay low (model correctly finds no structure) or grow unbounded (flag this as a pathological case — novelty threshold needs tuning).

**Figures:**
- `fig01a_sanity_accuracy.png` — bar chart of accuracy per task vs baselines
- `fig01b_uncertainty_by_task.png` — epistemic vs aleatoric var per task
- `fig01c_drift_epistemic_spike.png` — epistemic_var over time on drift task

---

## 6. Domain 02 — Regression

**Purpose:** Full regression benchmark against real-world continuous-target datasets.

**Data:** sklearn built-ins + UCI via ucimlrepo (zero/automatic download).

```python
# Data loading — no wget needed
from sklearn.datasets import load_diabetes, fetch_california_housing
from ucimlrepo import fetch_ucirepo

datasets = {
    'diabetes':           load_diabetes(return_X_y=True),
    'california_housing': fetch_california_housing(return_X_y=True),
    'concrete_strength':  _load_uci_concrete(),    # ID 165
    'energy_efficiency':  _load_uci_energy(),      # ID 242
    'wine_quality_red':   _load_uci_wine_red(),    # ID 186
    'auto_mpg':           _load_uci_auto_mpg(),    # ID 9
}

def _load_uci_concrete():
    ds = fetch_ucirepo(id=165)
    X = ds.data.features.values.astype(np.float32)
    y = ds.data.targets.values.flatten().astype(np.float32)
    return X, y
# Repeat pattern for other UCI datasets
```

**Protocol:**

For each dataset:
1. Standardise features (zero mean, unit variance) using training set statistics only
2. 80/20 train/test split (fixed random_state=42)
3. Train CyphaDIF online: iterate through training set one sample at a time (no batching)
4. Evaluate on test set: RMSE, MAE, R²
5. Record cypha_metrics() for epistemic/aleatoric split and α spectrum
6. Run all baselines in batch mode on identical train/test splits
7. Run `SGDRegressor` in online mode (same training loop) as the fairest comparison

**Additional regression-specific measurements:**

```python
# Predictive interval coverage
def coverage_at_confidence(model, X_test, y_test, z=1.96):
    """
    What fraction of test targets fall within the NIG predictive interval?
    Should be ≈ 0.95 for z=1.96 if calibration is correct.
    """
    covered = 0
    for x, y in zip(X_test, y_test):
        pred, ep_var, al_var = model.predict(x)
        total_std = np.sqrt(ep_var + al_var)
        if abs(y - pred) <= z * total_std:
            covered += 1
    return covered / len(y_test)
```

Expected: coverage at z=1.96 should be in [0.90, 0.99] on in-distribution data. Values outside this range indicate miscalibration.

**Figures:**
- `fig02a_regression_rmse_comparison.png` — RMSE per dataset, CyphaDIF vs all baselines
- `fig02b_predictive_interval_coverage.png` — coverage vs nominal confidence level (calibration curve)
- `fig02c_epistemic_vs_error.png` — scatter plot: epistemic_var vs absolute prediction error (should correlate)

---

## 7. Domain 03 — Classification

**Purpose:** Multi-class classification benchmark across tabular and encoded datasets.

**Data:** sklearn built-ins + 20 Newsgroups (auto-downloaded by sklearn, ~14MB).

```python
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                               load_digits, fetch_20newsgroups)
from sklearn.preprocessing import StandardScaler
from encoders.text_encoder import TextEncoder

datasets_classification = {
    'iris':         load_iris(return_X_y=True),       # 150 samples, 4 features, 3 classes
    'wine':         load_wine(return_X_y=True),        # 178 samples, 13 features, 3 classes
    'breast_cancer': load_breast_cancer(return_X_y=True),  # 569 samples, 30 features, 2 classes
    'digits_8x8':   load_digits(return_X_y=True),     # 1797 samples, 64 features, 10 classes
    '20newsgroups': _load_20news(),                    # 18846 samples, TF-IDF encoded, 20 classes
}

def _load_20news():
    data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    enc = TextEncoder(max_features=1000, ngram_range=(1,2))
    enc.fit(data.data)
    X = enc.encode_batch(data.data)
    return X, np.array(data.target)
```

**Protocol:**

For each dataset:
1. Standardise features on training split
2. 80/20 train/test split, stratified
3. Online training: one sample at a time, record accuracy, expert count, epistemic_var every 100 steps
4. Record final: accuracy, F1 (macro), confusion matrix
5. Record cypha_metrics()
6. Baselines: batch-trained on full training set

**Additional classification-specific measurements:**

```python
# Confidence-accuracy alignment
def confidence_accuracy_bins(model, X_test, y_test, n_bins=10):
    """
    Reliability diagram data for classification.
    Groups predictions by max routing probability (confidence),
    computes mean accuracy per bin.
    Perfect calibration: confidence == accuracy in each bin.
    """
    confidences, accuracies = [], []
    for x, y in zip(X_test, y_test):
        pred, probs, ep_var = model.predict(x)
        confidence = float(np.max(probs))
        correct = int(pred == y)
        confidences.append(confidence)
        accuracies.append(correct)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_confidences, bin_accuracies = [], []
    for i in range(n_bins):
        mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
        if mask.sum() > 0:
            bin_confidences.append(np.array(confidences)[mask].mean())
            bin_accuracies.append(np.array(accuracies)[mask].mean())
    return bin_confidences, bin_accuracies

# OOD detection via epistemic uncertainty
def ood_auroc(model, X_in, X_out):
    """
    AUROC for detecting OOD samples using epistemic_var as the score.
    Higher epistemic_var should indicate OOD.
    """
    from sklearn.metrics import roc_auc_score
    scores_in  = [model.predict(x)[2] for x in X_in]   # epistemic_var
    scores_out = [model.predict(x)[2] for x in X_out]
    labels = [0]*len(scores_in) + [1]*len(scores_out)
    scores = scores_in + scores_out
    return roc_auc_score(labels, scores)
```

For OOD test: train on `iris`, test OOD on random Gaussian noise. AUROC of epistemic_var should exceed 0.85.

**Figures:**
- `fig03a_classification_accuracy.png` — accuracy per dataset vs baselines
- `fig03b_reliability_diagram.png` — confidence-accuracy curve (all datasets overlaid)
- `fig03c_ood_auroc.png` — OOD detection AUROC for each dataset
- `fig03d_expert_count_vs_nclasses.png` — expert count at convergence vs number of classes

---

## 8. Domain 04 — Generation and Language Modelling

**Purpose:** Character-level and word-level next-token prediction on real text corpora.

**Data:**
- Gutenberg books (wget, Section 2.2): moby_dick.txt, sherlock_holmes.txt, alice.txt (~2MB total)
- WikiText-2 (wget, Section 2.2): word-level, standard LM benchmark (~4MB)

### 4.1 Character-Level Language Model

```python
# domains/d04_generation_language.py — Part A: Character-level

from encoders.text_encoder import CharNgramEncoder
import numpy as np

def run_char_lm(corpus_path, n=5, context_length=50, n_steps=50000, rng_seed=42):
    """
    Streams through the corpus character by character.
    At each position t, encodes the previous context_length characters as a
    char n-gram feature vector and predicts the next character.

    This tests CyphaDIF as a sequential density estimator.
    """
    with open(corpus_path, encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Build vocabulary from the text
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    # Build char n-gram encoder
    enc = CharNgramEncoder(n=n, vocab_size=min(200, vocab_size * 5))
    enc.build_vocab(text)

    # CyphaDIF as classifier over char vocabulary
    from cyphadif.classification import DIFClassifier
    from cyphadif.encoders import VectorEncoder
    clf = DIFClassifier(encoder=VectorEncoder(200), field_dim=160)

    # Training loop
    rng = np.random.default_rng(rng_seed)
    metrics = {'step': [], 'bits_per_char': [], 'epistemic_var': [],
               'expert_count': [], 'mean_alpha': []}

    window = text[:context_length]
    total_log_prob = 0.0
    n_trained = 0

    for i in range(context_length, min(len(text)-1, n_steps + context_length)):
        x = enc.encode(window)
        next_char = text[i]
        if next_char not in char2idx:
            window = window[1:] + next_char
            continue

        next_idx = char2idx[next_char]
        label = str(next_idx)

        # Train
        clf.train_step(x, label)

        # Evaluate every 500 steps
        if n_trained % 500 == 0:
            pred, probs, ep_var = clf.predict(x)
            # Bits per character: -log2(p(correct_char))
            true_prob = probs[next_idx] if next_idx < len(probs) else 1e-10
            bpc = -np.log2(max(true_prob, 1e-10))
            metrics['step'].append(n_trained)
            metrics['bits_per_char'].append(bpc)
            metrics['epistemic_var'].append(ep_var)
            metrics['expert_count'].append(clf.expert_count())
            metrics['mean_alpha'].append(clf.mean_alpha())

        window = window[1:] + next_char
        n_trained += 1

    return metrics

# Run on all three Gutenberg books
for book, path in [('moby_dick',      'data/gutenberg/moby_dick.txt'),
                   ('sherlock_holmes', 'data/gutenberg/sherlock_holmes.txt'),
                   ('alice',          'data/gutenberg/alice.txt')]:
    m = run_char_lm(path)
    # Save metrics for report
```

### 4.2 Word-Level Language Model on WikiText-2

```python
# Part B: WikiText-2 word-level
# Tokenise by whitespace, build TF-IDF context encoder over word n-grams

def load_wikitext2(data_dir='data/wikitext2/wikitext-2'):
    train_path = f'{data_dir}/wiki.train.tokens'
    test_path  = f'{data_dir}/wiki.test.tokens'
    with open(train_path) as f:
        train_text = f.read()
    with open(test_path) as f:
        test_text = f.read()
    return train_text, test_text

# Use context window of 20 words, TF-IDF encode the context
# Predict the next word (top-1000 vocabulary subset for tractability)
```

**Metrics:**
- Bits per character (BPC) for char-level; Perplexity for word-level
- BPC curve over training steps — should decrease monotonically
- Epistemic_var curve — should decrease as model compresses the corpus
- Expert count growth curve — should stabilise as model reaches saturation

**Baselines:**
- Unigram model BPC (frequency baseline)
- Bigram model BPC
- SGD-trained logistic regression over the same n-gram features

**Figure:**
- `fig04a_bpc_training_curves.png` — BPC over steps for each book
- `fig04b_wikitext2_perplexity.png` — Perplexity vs baseline
- `fig04c_expert_growth_language.png` — Expert count vs training steps on WikiText-2

---

## 9. Domain 05 — Game Theory: Chess

**Purpose:** CyphaDIF as a board evaluator — predicts game outcome from position features, then tests strategic expert specialisation.

**Data:** Kasparov.pgn (~2MB, downloaded in Section 2.2).

```python
# domains/d05_chess.py
import chess
import chess.pgn
import numpy as np
from encoders.chess_encoder import ChessEncoder

def load_pgn_positions(pgn_path, max_games=500, max_positions_per_game=40):
    """
    Parses PGN file. For each game, extracts board positions at each move.
    Returns (positions, outcomes) where outcome is +1 (white win), -1 (black win), 0 (draw).
    """
    enc = ChessEncoder()
    X, y = [], []

    with open(pgn_path, encoding='utf-8', errors='replace') as f:
        game_count = 0
        while game_count < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get('Result', '*')
            if result == '1-0':
                outcome = 1.0
            elif result == '0-1':
                outcome = -1.0
            elif result == '1/2-1/2':
                outcome = 0.0
            else:
                game_count += 1
                continue

            board = game.board()
            moves = list(game.mainline_moves())
            positions_this_game = 0

            for move in moves:
                if positions_this_game >= max_positions_per_game:
                    break
                board.push(move)
                features = enc.encode(board)
                X.append(features)
                y.append(outcome)
                positions_this_game += 1

            game_count += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
```

**Experiments:**

**Experiment 5A — Position Outcome Regression:**
Train CyphaDIF to predict game outcome (+1/-1/0) from board features. This tests whether position evaluation correlates with actual results.
- Metric: RMSE on held-out positions, R², uncertainty calibration
- Baseline: Ridge regression, Random Forest

**Experiment 5B — Move Quality Classification:**
Categorise positions as "winning", "equal", or "losing" (from outcome + side to move).
- Metric: 3-class accuracy, confusion matrix
- Expected: CyphaDIF experts should specialise into opening/middlegame/endgame positions naturally

**Experiment 5C — Expert Specialisation by Game Phase:**
After training, inspect which expert activates most strongly for:
- Opening positions (< 10 moves played)
- Middlegame (10-30 moves)
- Endgame (< 7 total pieces remaining)
Plot routing probability distributions for each phase.

**Experiment 5D — Tactical vs Positional:**
Label positions as tactical (forced sequence exists) or positional (no immediate tactic).
Heuristic: a position is "tactical" if the piece mobility difference exceeds a threshold.
Does CyphaDIF route tactical and positional positions to different experts?

**Experiment 5E — Uncertainty in Novel Positions:**
Feed the model positions from a different player (non-Kasparov) not seen in training.
Epistemic_var should be higher on these novel positions than on Kasparov's typical styles.

**Figure:**
- `fig05a_chess_rmse_vs_baselines.png`
- `fig05b_expert_phase_routing.png` — heatmap of expert activation by game phase
- `fig05c_chess_uncertainty_novel.png` — epistemic_var distribution: Kasparov vs novel player

---

## 10. Domain 06 — Game Theory: Go

**Purpose:** CyphaDIF as a territory estimator and move strength classifier on 9×9 Go.

**Data:** Synthetic — 9×9 positions generated programmatically (no download needed). Optionally load KGS 9×9 SGF files if available.

```python
# domains/d06_go.py
import numpy as np
from encoders.go_encoder import GoEncoder

def generate_synthetic_go_position(rng: np.random.Generator, n_stones=20):
    """
    Generate a plausible-ish 9×9 board by randomly placing stones.
    Returns (board_array, territory_differential) where territory is 
    simply the stone differential (approximation).
    """
    board = np.zeros((9, 9), dtype=np.float32)
    positions = rng.choice(81, size=n_stones, replace=False)
    for i, pos in enumerate(positions):
        r, c = divmod(pos, 9)
        board[r, c] = 1.0 if i < n_stones // 2 else -1.0
    territory = float(board.sum())  # Positive = Black advantage
    return board, territory

def generate_dataset(n_samples=10000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    enc = GoEncoder()
    X, y_reg, y_cls = [], [], []

    for _ in range(n_samples):
        n_stones = int(rng.integers(5, 40))
        board, territory = generate_synthetic_go_position(rng, n_stones)
        features = enc.encode(board)
        X.append(features)
        y_reg.append(territory)
        # Classification: black winning (+1), white winning (-1), close (0)
        if territory > 3:
            y_cls.append('black')
        elif territory < -3:
            y_cls.append('white')
        else:
            y_cls.append('close')

    return np.array(X), np.array(y_reg), y_cls
```

**Experiments:**

**Experiment 6A — Territory Estimation (Regression):**
Predict the territory differential from board features.
- Metric: RMSE, MAE, R²
- Baseline: Ridge, Random Forest

**Experiment 6B — Outcome Classification:**
3-class: black winning / close / white winning.
- Metric: accuracy, F1

**Experiment 6C — Board Density Expert Specialisation:**
Hypothesis: experts should specialise by board density (early game = sparse, late game = dense).
Measure: plot mean stone count vs dominant expert index.

**Experiment 6D — 9×9 vs 5×5 Distribution Shift:**
Train on 9×9 positions. Then present 5×5 positions encoded into the same 168-dim space (padded with zeros). Epistemic_var should be higher on 5×5 — model correctly signals OOD.

**Figure:**
- `fig06a_go_territory_regression.png`
- `fig06b_go_expert_by_density.png`
- `fig06c_go_ood_detection.png`

---

## 11. Domain 07 — Game Theory: Poker

**Purpose:** CyphaDIF as a hand evaluator and decision classifier (fold/call/raise). Tests the model on high-uncertainty, adversarial decision-making.

**Data:** Synthetic — generated by `PokerEncoder.generate_random_situation()`. No download needed. The `treys` library does the hand evaluation.

```python
# domains/d07_poker.py
import numpy as np
from encoders.poker_encoder import PokerEncoder

def generate_poker_dataset(n_hands=20000, rng_seed=42):
    """Generate synthetic Texas Hold'em situations with ground-truth decisions."""
    enc = PokerEncoder()
    rng = np.random.default_rng(rng_seed)
    X, y = [], []

    for _ in range(n_hands):
        vec, label = enc.generate_random_situation(rng)
        X.append(vec)
        y.append(label)  # 'fold', 'call', 'raise'

    return np.array(X), y
```

**Experiments:**

**Experiment 7A — Decision Classification (fold/call/raise):**
- Metric: 3-class accuracy, confusion matrix, F1 per class
- Baseline: Random Forest, k-NN

**Experiment 7B — Uncertainty on Borderline Hands:**
Hands near the fold/call boundary (hand_strength ≈ 0.35) should have HIGHER epistemic_var than strong or weak hands. This is the most strategically valuable property — the model knows when it's uncertain.
- Metric: Spearman correlation of epistemic_var with |hand_strength - 0.35| (should be NEGATIVE — closer to boundary = more uncertain)

**Experiment 7C — EV Regression:**
Regress expected value (computed as hand_strength × pot_size − (1 − hand_strength) × call_size) from features.
- Metric: RMSE, coverage

**Experiment 7D — Expert Specialisation by Street:**
Hypothesis: separate experts should activate for pre-flop, flop, turn, and river decisions. Plot mean street feature vs dominant expert index.

**Experiment 7E — Position Awareness:**
Does CyphaDIF learn that the same hand is worth more in late position? Test: hold hand features constant, vary position feature. Measure prediction change.

**Figure:**
- `fig07a_poker_decision_accuracy.png`
- `fig07b_poker_uncertainty_boundary.png` — epistemic_var vs distance to fold/call boundary
- `fig07c_poker_expert_by_street.png`

---

## 12. Domain 08 — Computer Vision

**Purpose:** CyphaDIF on image data. Tests three encoding strategies (raw pixels, HOG, patch routing) against MNIST.

**Data:** MNIST (11MB, wget in Section 2.2).

```python
# domains/d08_computer_vision.py
import numpy as np, gzip, struct
from encoders.image_encoder import ImageEncoder

def load_mnist(data_dir='data/mnist'):
    def read_images(path):
        with open(path, 'rb') as f:
            magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    def read_labels(path):
        with open(path, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images(f'{data_dir}/train-images-idx3-ubyte')
    y_train = read_labels(f'{data_dir}/train-labels-idx1-ubyte')
    X_test  = read_images(f'{data_dir}/t10k-images-idx3-ubyte')
    y_test  = read_labels(f'{data_dir}/t10k-labels-idx1-ubyte')
    return X_train, y_train, X_test, y_test
```

**Experiments:**

**Experiment 8A — Raw Pixel Classification (784 features):**
Train on 10k MNIST samples, test on 2k. Baseline reference point.
- Metric: accuracy, F1 per digit

**Experiment 8B — HOG Feature Classification (~144 features):**
Same protocol with HOG-encoded images. HOG is much lower-dimensional — tests whether structural features are more useful per dimension for CyphaDIF.

**Experiment 8C — Patch Routing:**
Each 28×28 image is split into 16 patches (4×4 patches of 7×7 pixels each). Each patch is independently routed through a CyphaDIF instance. Final prediction is a majority vote of the 16 patch predictions weighted by (1 − epistemic_var_patch).
```
Aggregated prediction: argmax of weighted sum of per-patch routing probabilities
Weight of patch k: 1 / (1 + epistemic_var_k)
```
- Metric: accuracy vs single-encoder approach
- Novel measurement: which patches are most uncertain (epistemic_var)? Visualise as a 4×4 uncertainty map averaged over the test set. The background patches should be MORE uncertain than the digit patches.

**Experiment 8D — Class Specialisation:**
After training on all 10 digits, inspect expert_means for the top-10 experts. Each expert's mean feature vector can be decoded back to an 8×8 image (for HOG encoding) or displayed as a prototypical pattern.
- Expected: visual prototypes should be recognisable digit-like patterns

**Experiment 8E — Online Digit Streaming:**
Stream 60k MNIST training images one at a time in class-sorted order (all 0s, then all 1s, etc.). This creates severe concept drift. Plot:
- Accuracy on held-out test set after each class batch
- Expert count over time
- Epistemic_var on each new class vs seen classes

**Experiment 8F — MNIST OOD:**
Train on digits 0-7. Present digits 8 and 9 as OOD. AUROC of epistemic_var for OOD detection. Expected: > 0.75.

**Figure:**
- `fig08a_mnist_accuracy_by_encoding.png`
- `fig08b_patch_uncertainty_map.png` — average epistemic_var per patch position
- `fig08c_expert_visual_prototypes.png` — decoded expert means
- `fig08d_streaming_class_experiment.png` — accuracy + epistemic_var over class-sorted stream

---

## 13. Domain 09 — Document Understanding

**Purpose:** Topic classification, streaming document routing, and cross-document uncertainty.

**Data:**
- 20 Newsgroups (sklearn auto-download, ~14MB): 18,846 posts, 20 classes
- Gutenberg texts (wget, Section 2.2): streaming document test

### 9.1 20 Newsgroups Classification

```python
from sklearn.datasets import fetch_20newsgroups
from encoders.text_encoder import TextEncoder

def load_20news():
    train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
    test  = fetch_20newsgroups(subset='test',  remove=('headers','footers','quotes'))
    enc   = TextEncoder(max_features=2000, ngram_range=(1, 2))
    enc.fit(train.data)
    X_train = enc.encode_batch(train.data)
    X_test  = enc.encode_batch(test.data)
    return X_train, train.target, X_test, test.target, train.target_names
```

**Experiments:**

**Experiment 9A — 20-Class Topic Classification:**
- Metric: accuracy, macro-F1
- Baseline: TF-IDF + Logistic Regression (strong baseline for this task), SGDClassifier

**Experiment 9B — Topic Expert Specialisation:**
After training, compare the expert routing distributions for topically similar categories:
- `comp.graphics` vs `comp.os.ms-windows.misc` vs `comp.sys.ibm.pc.hardware` — should route to similar experts
- `rec.sport.hockey` vs `rec.sport.baseball` — same sport expert?
- `alt.atheism` vs `talk.religion.misc` — should they share an expert?
Plot a 20×K expert routing heatmap to visualise topic→expert mapping.

**Experiment 9C — Cross-Domain OOD (Newsgroups vs Gutenberg):**
Train on 20 Newsgroups. Encode Gutenberg text segments (1000-char windows) with the same TF-IDF encoder. Epistemic_var on Gutenberg should be significantly higher than on held-out newsgroup posts.
- Metric: Mann-Whitney U test for significance of epistemic_var difference

**Experiment 9D — Document Streaming with Topic Drift:**
Stream newsgroup posts in topic-sorted order. Simulate a document stream that switches topics every 1000 posts. Plot epistemic_var over the stream — should spike at topic boundaries.

### 9.2 Gutenberg Passage Classification

```python
def segment_book(path, segment_chars=500):
    """Split a book into fixed-length text segments for classification."""
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    segments = [text[i:i+segment_chars] for i in range(0, len(text)-segment_chars, segment_chars)]
    return segments

# 3-class problem: which book does this passage come from?
# Alice (~360 segments), Sherlock (~1200 segments), Moby Dick (~2400 segments)
# TF-IDF encode all segments, then train/test on 80/20 split
```

**Figure:**
- `fig09a_20news_accuracy.png`
- `fig09b_topic_expert_heatmap.png` — 20 classes × K experts routing matrix
- `fig09c_gutenberg_ood_epistemic.png`
- `fig09d_topic_drift_stream.png`

---

## 14. Domain 10 — Time Series

**Purpose:** Classification and regression on real physiological and financial time series.

**Data:**
- ECG5000 (wget, Section 2.2): 5000 ECG sequences, 5 classes, 140 timesteps
- Financial: Yahoo Finance via yfinance (live pull, tiny data)

### 10.1 ECG5000 Classification

```python
# domains/d10_time_series.py
import numpy as np
import pandas as pd
from encoders.timeseries_encoder import TimeSeriesEncoder

def load_ecg5000():
    train = np.loadtxt('data/ecg5000/ECG5000_TRAIN.txt')
    test  = np.loadtxt('data/ecg5000/ECG5000_TEST.txt')
    # First column is label, remaining 140 are the time series
    X_train, y_train = train[:, 1:], train[:, 0].astype(int)
    X_test,  y_test  = test[:,  1:], test[:,  0].astype(int)
    return X_train, y_train, X_test, y_test

# Each row is already a full 140-step time series.
# Encode each row with TimeSeriesEncoder (treating the full series as one window).
# Also: sliding window experiment within each series (10-step windows, 130 windows per series).
```

**Experiments:**

**Experiment 10A — ECG Beat Classification (5-class):**
Encode each 140-step ECG as statistical + FFT features. Train online, test on held-out set.
- Classes: Normal, R-on-T PVC, PVC, SP/EB, Unclassified
- Metric: 5-class accuracy, per-class F1
- Key test: epistemic_var should be highest for class 5 (Unclassified/ambiguous) — the model correctly identifies the uncertain class

**Experiment 10B — ECG Sliding Window:**
For each ECG series, extract 10-step sliding windows (120 windows per series). Train on the statistical features of these windows to predict the series' class. Tests whether CyphaDIF can classify from partial observations.

**Experiment 10C — Arrhythmia OOD:**
Train only on Normal ECGs. Test epistemic_var on arrhythmia classes. Should be detectably higher.
- Metric: AUROC for binary OOD detection (normal vs arrhythmia)

### 10.2 Financial Time Series

```python
import yfinance as yf

def load_financial_data(tickers=['SPY','QQQ','GLD','TLT'], period='5y', interval='1d'):
    """
    Download daily price data. Returns dict of DataFrames.
    yfinance caches locally so subsequent runs are instant.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        data[ticker] = df
    return data

def build_return_features(df, window=20):
    """
    From daily prices, build features: rolling returns, volatility, momentum.
    Target: next-day return sign (up=1, down=0).
    """
    enc = TimeSeriesEncoder(window_size=window)
    prices = df['Close'].values
    log_returns = np.diff(np.log(prices))
    X, indices = enc.sliding_windows(log_returns, step=1)
    y = (log_returns[indices] > 0).astype(int)
    return X, y
```

**Experiments:**

**Experiment 10D — Return Sign Prediction:**
Binary classification: will the market go up or down tomorrow?
- Tickers: SPY, QQQ, GLD, TLT (broad market, tech, gold, bonds)
- Metric: accuracy, AUROC (note: near-50% expected — financial data is hard)
- Key test: CyphaDIF should output HIGH epistemic_var on all financial predictions, correctly signalling it cannot reliably predict — this is the honest uncertainty behaviour.

**Experiment 10E — Volatility Regime Detection:**
Cluster market periods by rolling volatility. High-vol periods (crisis) vs low-vol (bull market). Does CyphaDIF route crisis-period windows to different experts than calm-period windows?
- Label high-vol periods as those where 20-day rolling std > 1.5× its annual mean.

**Experiment 10F — Cross-Asset Transfer:**
Train on SPY (US equities), then test on GLD (gold). Epistemic_var on GLD should initially be high, decreasing as the model adapts online to the different return distribution.

**Figure:**
- `fig10a_ecg_classification_accuracy.png`
- `fig10b_ecg_ood_detection.png`
- `fig10c_financial_uncertainty_by_regime.png`
- `fig10d_expert_specialisation_by_volatility.png`

---

## 15. Domain 11 — Reinforcement Learning and Reward Modelling

**Purpose:** CyphaDIF as a value function estimator and reward model in simple RL environments. Tests whether uncertainty-aware predictions improve policy evaluation.

**Data:** Synthetic RL trajectories generated from simple environments (no OpenAI Gym required — pure numpy implementations).

```python
# domains/d11_reinforcement_learning.py
import numpy as np

class CartPoleEnv:
    """
    Minimal CartPole physics in pure numpy.
    State: [cart_pos, cart_vel, pole_angle, pole_vel] — 4 features.
    Action: 0 (push left) or 1 (push right).
    Episode terminates when |pole_angle| > 12° or |cart_pos| > 2.4.
    """
    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)
        self.g, self.mc, self.mp, self.l = 9.8, 1.0, 0.1, 0.5
        self.dt, self.force = 0.02, 10.0
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force if action == 1 else -self.force
        costheta, sintheta = np.cos(theta), np.sin(theta)
        tmp = (force + self.mp * self.l * theta_dot**2 * sintheta) / (self.mc + self.mp)
        theta_acc = (self.g * sintheta - costheta * tmp) / \
                    (self.l * (4/3 - self.mp * costheta**2 / (self.mc + self.mp)))
        x_acc = tmp - self.mp * self.l * theta_acc * costheta / (self.mc + self.mp)
        self.state = np.array([
            x + self.dt * x_dot,
            x_dot + self.dt * x_acc,
            theta + self.dt * theta_dot,
            theta_dot + self.dt * theta_acc
        ])
        done = (abs(self.state[2]) > 0.2094) or (abs(self.state[0]) > 2.4)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done

class GridWorldEnv:
    """4×4 GridWorld. State: (row, col) encoded as 16-dim one-hot. Reward: +1 at goal, 0 elsewhere."""
    def __init__(self, size=4, rng_seed=42):
        self.size = size
        self.goal = (size-1, size-1)
        self.rng  = np.random.default_rng(rng_seed)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._encode()

    def step(self, action):
        moves = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        dr, dc = moves[action]
        r = max(0, min(self.size-1, self.pos[0] + dr))
        c = max(0, min(self.size-1, self.pos[1] + dc))
        self.pos = (r, c)
        done   = (self.pos == self.goal)
        reward = 1.0 if done else 0.0
        return self._encode(), reward, done

    def _encode(self):
        vec = np.zeros(self.size * self.size, dtype=np.float32)
        vec[self.pos[0] * self.size + self.pos[1]] = 1.0
        return vec
```

**Experiments:**

**Experiment 11A — Value Function Regression (CartPole):**
Run 1000 random-policy episodes. Record (state, discounted_return) pairs. Train CyphaDIF to predict the discounted return from the state.
- Metric: RMSE, R² vs Ridge baseline
- Key test: states where the pole is near-tipping (|theta| ≈ 0.15 rad) should have HIGH aleatoric_var (inherently noisy outcomes) AND high epistemic_var early in training (model hasn't seen many near-tipping states).

**Experiment 11B — Policy Evaluation (GridWorld):**
Run ε-greedy policy. CyphaDIF estimates Q(s,a) for each state-action pair.
- Compare estimated Q-values to true Q-values (computable analytically for 4×4 grid with γ=0.9)
- Metric: MAE between estimated and true Q-values

**Experiment 11C — Reward Model (Preference Learning):**
Generate pairs of trajectories from the same start state. Label which trajectory got higher total reward. Train CyphaDIF to classify trajectory preference from trajectory summary features (mean state, total steps, variance of states).
- Metric: pairwise preference accuracy
- Baseline: Bradley-Terry model fitted with logistic regression

**Experiment 11D — Epistemic Uncertainty for Exploration:**
In a tabular RL setting (GridWorld), use CyphaDIF's epistemic_var as the exploration bonus. States with high epistemic_var (not yet seen) get a bonus reward. Compare cumulative reward curve vs ε-greedy exploration over 200 episodes.

**Figure:**
- `fig11a_value_function_quality.png`
- `fig11b_qvalue_estimation.png`
- `fig11c_uncertainty_exploration_bonus.png` — cumulative reward: epistemic bonus vs ε-greedy

---

## 16. Domain 12 — Anomaly and OOD Detection

**Purpose:** CyphaDIF's epistemic uncertainty as an anomaly score. Tested on the NSL-KDD network intrusion dataset.

**Data:** NSL-KDD (wget, Section 2.2): network connection records, binary label (normal vs attack).

```python
# domains/d12_anomaly_detection.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

NSL_KDD_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
]

def load_nsl_kdd():
    train = pd.read_csv('data/nsl_kdd/KDDTrain+.txt', header=None, names=NSL_KDD_COLS)
    test  = pd.read_csv('data/nsl_kdd/KDDTest+.txt',  header=None, names=NSL_KDD_COLS)

    # Encode categorical features
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    # Binary label: normal=0, attack=1
    train['binary_label'] = (train['label'] != 'normal').astype(int)
    test['binary_label']  = (test['label']  != 'normal').astype(int)

    feature_cols = [c for c in NSL_KDD_COLS if c not in ('label','difficulty_level')]
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train['binary_label'].values
    X_test  = test[feature_cols].values.astype(np.float32)
    y_test  = test['binary_label'].values

    # Attack type labels for fine-grained experiment
    attack_types = test['label'].values

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test, attack_types
```

**Experiments:**

**Experiment 12A — Binary Intrusion Detection:**
Train on normal traffic only (unsupervised setup). Use epistemic_var as the anomaly score. Test on held-out mix of normal and attack traffic.
- Metric: AUROC, F1 at optimal threshold, precision-recall curve
- Baseline: Isolation Forest, One-Class SVM, LOF

**Experiment 12B — Attack Type Discrimination:**
Train on 4 main attack types (DoS, Probe, R2L, U2R) as separate classes. Test whether CyphaDIF experts specialise by attack type. After training, feed novel attack subtypes not seen in training — epistemic_var should be higher on novel subtypes.

**Experiment 12C — Online Detection (Streaming):**
Stream test connections in chronological order. After each connection, CyphaDIF updates online. Measure detection latency: how many examples of a new attack type does CyphaDIF need before its accuracy on that type exceeds 0.80?

**Experiment 12D — Adaptive Threshold:**
Standard detectors use a fixed threshold. CyphaDIF's epistemic_var naturally provides a confidence-adaptive threshold: only flag as anomaly if epistemic_var > threshold AND routing probability to "attack" expert > 0.5. Compare false positive rate vs fixed-threshold approach.

**Figure:**
- `fig12a_nslkdd_roc_curve.png`
- `fig12b_attack_type_expert_routing.png`
- `fig12c_online_detection_latency.png`
- `fig12d_adaptive_threshold_fp_reduction.png`

---

## 17. Domain 13 — Compression

**Purpose:** CyphaDIF's GRIA α-parameter as a compression oracle. Tests whether the Grand Unified Law correctly identifies compressible vs incompressible data.

**Data:** Canterbury Corpus (wget, Section 2.2): standard compression benchmark files.

```python
# domains/d13_compression.py
import os
import numpy as np
from encoders.text_encoder import CharNgramEncoder

CANTERBURY_FILES = {
    'alice29.txt':    'literary prose',
    'asyoulik.txt':   'Shakespeare play',
    'cp.html':        'HTML markup',
    'fields.c':       'C source code',
    'grammar.lsp':    'Lisp code',
    'kennedy.xls':    'binary spreadsheet',
    'lcet10.txt':     'technical writing',
    'plrabn12.txt':   'poetry',
    'ptt5':           'binary fax',
    'sum':            'binary executable',
    'xargs.1':        'Unix man page',
}
```

**Experiments:**

**Experiment 13A — Compressibility vs α (Grand Unified Law):**
For each Canterbury file:
1. Encode the file as a stream of 5-char n-gram feature vectors
2. Train CyphaDIF on the stream for 10k steps
3. Record mean_alpha at convergence
4. Compute the actual compression ratio achievable by gzip, zstd (baseline compressors) on the same file

Hypothesis: files where CyphaDIF converges to high α (more irreversible = more structure found) should also be more compressible by gzip. The GUL α should correlate with actual compression ratio.
- Metric: Spearman correlation between mean_alpha and gzip compression ratio across files
- Expected: positive correlation > 0.6

```python
import gzip, os

def compute_compression_ratio(filepath):
    with open(filepath, 'rb') as f:
        original = f.read()
    compressed = gzip.compress(original, compresslevel=9)
    return len(original) / len(compressed)
```

**Experiment 13B — Binary vs Text α:**
Binary files (kennedy.xls, ptt5, sum) should have lower gzip compressibility AND lower CyphaDIF α at convergence (random byte streams have no extractable structure for a char n-gram model).
Text files should have higher compressibility AND higher α.
Test this split statistically.

**Experiment 13C — Code vs Prose Specialisation:**
Does CyphaDIF route C code (fields.c) and Lisp code (grammar.lsp) to different experts than literary prose (alice29.txt, plrabn12.txt)? If yes: the model has discovered the code vs prose distinction without labels — expert specialisation by data type.

**Experiment 13D — Entropy Estimation Accuracy:**
Estimate entropy per character of each file using CyphaDIF's predictions:
`H_est = -mean(log2(p(correct_char)))` over the stream.
Compare against true entropy estimated by a large-order context model.
- Metric: MAE between CyphaDIF entropy estimate and reference entropy

**Figure:**
- `fig13a_gul_alpha_vs_compression_ratio.png` — scatter plot: α vs gzip ratio per file
- `fig13b_binary_vs_text_alpha.png`
- `fig13c_code_prose_expert_routing.png`

---

## 18. Domain 14 — Scientific and Symbolic Regression

**Purpose:** CyphaDIF as a scientific function approximator. Tests whether it can fit relationships from known physics/engineering equations without knowing the functional form.

**Data:** Feynman Symbolic Regression Benchmark — individual equations as small CSV files. Generate them in-script using the known equations (no external data needed for clean benchmarks).

```python
# domains/d14_symbolic_regression.py
import numpy as np

# 20 representative Feynman equations, generated synthetically
EQUATIONS = {
    'newton_second_law':   lambda F, m: F / m,                     # a = F/m
    'kinetic_energy':      lambda m, v: 0.5 * m * v**2,           # KE = 0.5mv²
    'gravitational_pe':    lambda m, g, h: m * g * h,              # PE = mgh
    'ohms_law':            lambda V, R: V / R,                     # I = V/R
    'ideal_gas':           lambda n, R, T, V: (n * R * T) / V,     # P = nRT/V
    'coulombs_law':        lambda q1, q2, r: (8.99e9 * q1 * q2) / r**2,
    'wave_speed':          lambda lam, f: lam * f,
    'relativistic_KE':     lambda m, v, c: m * c**2 * (1/np.sqrt(1-(v/c)**2) - 1),
    'lens_equation':       lambda do, di: 1/(1/do + 1/di),         # f = (do*di)/(do+di)
    'bernoulli':           lambda rho, v, P: P + 0.5 * rho * v**2,
    'hooke':               lambda k, x: 0.5 * k * x**2,
    'snell':               lambda n1, theta1, n2: n1 * np.sin(theta1) / n2,
    'Stefan_Boltzmann':    lambda sigma, T: sigma * T**4,
    'thermal_expansion':   lambda L0, alpha, dT: L0 * alpha * dT,
    'capacitor_energy':    lambda C, V: 0.5 * C * V**2,
    'log_decay':           lambda N0, lam, t: N0 * np.exp(-lam * t),
    'centripetal':         lambda m, v, r: m * v**2 / r,
    'diffraction':         lambda lam, d: np.arcsin(lam / d),
    'entropy_ideal_gas':   lambda n, Cv, T: n * Cv * np.log(T),
    'drag_force':          lambda Cd, rho, A, v: 0.5 * Cd * rho * A * v**2,
}

def generate_feynman_dataset(equation_fn, n_samples=2000, noise_std=0.01, rng_seed=42):
    """
    Generate (X, y) pairs for a given equation.
    All inputs are drawn from [0.1, 10] (avoiding singularities near 0).
    Adds small Gaussian noise to simulate measurement error.
    """
    rng = np.random.default_rng(rng_seed)
    import inspect
    n_inputs = len(inspect.signature(equation_fn).parameters)
    X = rng.uniform(0.1, 5.0, (n_samples, n_inputs))
    y = np.array([equation_fn(*row) for row in X], dtype=np.float32)
    y += rng.standard_normal(n_samples) * noise_std * (np.abs(y).mean() + 1e-8)
    return X.astype(np.float32), y
```

**Experiments:**

**Experiment 14A — Fitting All 20 Equations (Online Regression):**
For each equation, generate 2000 samples. Train CyphaDIF online on 1600, test on 400.
- Metric: RMSE, R², normalised RMSE (NRMSE = RMSE / std(y))
- Baseline: Ridge, Random Forest, Gradient Boosting

**Experiment 14B — Extrapolation Uncertainty:**
Train on inputs in [0.1, 5.0]. Test on inputs in [5.1, 10.0] — outside training range.
- Epistemic_var should be dramatically higher outside the training range.
- Metric: AUROC using epistemic_var to detect extrapolation (inputs [0.1, 5.0] = in-distribution, [5.1, 10.0] = OOD)
- Expected: AUROC > 0.80

**Experiment 14C — Noisy vs Clean Signal:**
For kinetic_energy and ohm's_law, vary noise_std from 0 to 0.5. Plot:
- Aleatoric_var vs noise_std (should correlate strongly — aleatoric captures measurement noise)
- RMSE vs noise_std (degradation curve)

**Experiment 14D — Expert Specialisation by Functional Form:**
Train CyphaDIF on a mixture of 5 equations simultaneously (kinetic_energy, ohm's_law, gravitational_pe, hooke, log_decay). Label each sample by its source equation. After training, measure: do samples from the same equation route to the same expert?
Metric: Adjusted Rand Index between expert routing and true equation label.
Expected: ARI > 0.5 — CyphaDIF discovers the functional structure.

**Figure:**
- `fig14a_feynman_benchmark_rmse.png` — RMSE heatmap: 20 equations × model variants
- `fig14b_extrapolation_uncertainty.png`
- `fig14c_aleatoric_vs_noise.png`
- `fig14d_equation_expert_specialisation.png`

---

## 19. Domain 15 — Adversarial Robustness

**Purpose:** How does CyphaDIF respond to adversarial inputs, distribution shifts, and deliberate corruption? Tests whether epistemic_var is a reliable signal for adversarial detection.

**Data:** MNIST (already downloaded), 20 Newsgroups (already downloaded), synthetic corruptions.

```python
# domains/d15_adversarial_robustness.py
import numpy as np

def add_gaussian_noise(X, std, rng):
    return X + rng.standard_normal(X.shape) * std

def add_salt_and_pepper(X, fraction, rng):
    X_noisy = X.copy()
    mask = rng.random(X.shape) < fraction
    X_noisy[mask] = rng.choice([0.0, 1.0], size=mask.sum())
    return X_noisy

def feature_dropout(X, dropout_rate, rng):
    mask = rng.random(X.shape) < dropout_rate
    X_dropped = X.copy()
    X_dropped[mask] = 0.0
    return X_dropped

def adversarial_fgsm_proxy(model, x, y_true, epsilon=0.1):
    """
    Approximate FGSM-style perturbation without gradients:
    perturb each feature by epsilon in the direction that increases prediction error.
    Uses finite differences.
    """
    x_adv = x.copy()
    pred_orig, probs_orig, _ = model.predict(x)

    for i in range(len(x)):
        x_plus  = x.copy(); x_plus[i]  += 1e-4
        x_minus = x.copy(); x_minus[i] -= 1e-4
        p_plus,  _, _ = model.predict(x_plus)
        p_minus, _, _ = model.predict(x_minus)

        # Move in the direction that increases cross-entropy on true class
        # (gradient sign approximation)
        if p_plus != y_true:
            x_adv[i] += epsilon
        elif p_minus != y_true:
            x_adv[i] -= epsilon

    return np.clip(x_adv, 0.0, 1.0)
```

**Experiments:**

**Experiment 15A — Gaussian Noise Robustness:**
Train CyphaDIF on MNIST HOG features. Test on HOG features with increasing Gaussian noise (std = 0.0, 0.1, 0.2, 0.5, 1.0).
- Metric: accuracy vs noise level
- Hypothesis: epistemic_var should increase with noise level, signalling degraded confidence

**Experiment 15B — Feature Dropout:**
Randomly zero out 10%, 25%, 50%, 75% of features at test time.
- Metric: accuracy vs dropout rate
- Hypothesis: high dropout → high epistemic_var

**Experiment 15C — Adversarial Perturbation (FGSM proxy):**
Apply `adversarial_fgsm_proxy` to 500 test MNIST images. Compare:
- Accuracy before and after perturbation
- Epistemic_var before and after perturbation
- Hypothesis: adversarial examples should have HIGHER epistemic_var than natural inputs — the model is entering unexplored feature space, which it correctly signals as uncertain.

**Experiment 15D — Label Noise Training:**
Introduce 0%, 10%, 20%, 30% random label noise during training. Measure:
- Test accuracy vs label noise level
- Expert count vs label noise (noisy labels should generate more experts as the model tries to separate contradictory examples)
- Aleatoric_var on noisy vs clean training labels — should be detectably higher on noisy labels

**Experiment 15E — Temporal Distribution Shift (Gradual):**
Gradually rotate the feature space over 5000 training steps (apply a slow rotation matrix to X). Plot:
- Accuracy over time
- Epistemic_var over time (should spike as the distribution drifts, then recover as the model adapts)
- Expert creation events (new experts should be created at peak drift)

**Figure:**
- `fig15a_noise_robustness.png` — accuracy + epistemic_var vs noise level
- `fig15b_adversarial_epistemic.png` — epistemic_var distribution: natural vs adversarial
- `fig15c_label_noise_expert_count.png`
- `fig15d_gradual_drift_response.png`

---

## 20. Domain 16 — Multi-Task Simultaneous Learning

**Purpose:** CyphaDIF handling multiple tasks simultaneously through a single model. Tests whether experts specialise by task without explicit task labels.

**Data:** Mix of already-loaded datasets: UCI regression + 20 Newsgroups classification + ECG classification, run simultaneously.

```python
# domains/d16_multitask.py
import numpy as np
from itertools import cycle

def multitask_stream(task_datasets, interleave='round_robin', rng_seed=42):
    """
    Generator that yields (features, label, task_id) tuples from multiple datasets.
    In 'round_robin' mode: alternates between tasks.
    In 'random' mode: picks a random task at each step.
    In 'block' mode: 1000 steps per task, then switches.
    """
    rng = np.random.default_rng(rng_seed)
    iterators = {tid: cycle(zip(X, y)) for tid, (X, y) in task_datasets.items()}
    task_ids  = list(task_datasets.keys())

    while True:
        if interleave == 'round_robin':
            for tid in task_ids:
                x, y = next(iterators[tid])
                yield x, y, tid
        elif interleave == 'random':
            tid = rng.choice(task_ids)
            x, y = next(iterators[tid])
            yield x, y, tid
        elif interleave == 'block':
            for tid in task_ids:
                for _ in range(1000):
                    x, y = next(iterators[tid])
                    yield x, y, tid
```

**Experiments:**

**Experiment 16A — Task Identity Discovery:**
Train a single CyphaDIF on 3 tasks simultaneously (round-robin), with task IDs hidden. After 50k steps, cluster the expert routing probabilities using k-means. Do the discovered clusters match the true task labels?
- Metric: Adjusted Rand Index between expert cluster assignment and true task label
- Expected: ARI > 0.5 — tasks should route to different expert clusters

**Experiment 16B — Catastrophic Forgetting Resistance:**
Protocol: Train on Task A (10k steps) → train on Task B (10k steps) → train on Task C (10k steps) → re-evaluate on Task A.
Compare Task A accuracy before vs after Task B and C training.
Compare against fine-tuned Random Forest (which will forget A completely) and against SGD (which catastrophically forgets).

**Experiment 16C — Task Complexity vs Expert Count:**
Use 1 simple task (iris, 4 features, 3 classes) and 1 complex task (20 Newsgroups, 2000 features, 20 classes) simultaneously. Measure expert count allocated to each task by inspecting which task's samples dominate each expert.
Hypothesis: complex task allocates more experts than simple task.

**Experiment 16D — Interleaving Strategy Comparison:**
Compare round_robin vs random vs block interleaving on the same 3-task setup.
- Metric: final accuracy on all 3 tasks after identical total steps
- Block interleaving is the hardest for catastrophic forgetting — does CyphaDIF handle it?

**Figure:**
- `fig16a_multitask_task_discovery.png` — t-SNE of expert routing coloured by task
- `fig16b_forgetting_resistance_bar.png` — Task A accuracy: before vs after Tasks B,C
- `fig16c_expert_allocation_by_task.png`
- `fig16d_interleaving_strategy_comparison.png`

---

## 21. Domain 17 — CyphaLM Integration

**Purpose:** Test CyphaLM (the LLM-capable architecture being built now) through the same rigorous bench once its components are available. This domain is written against the CyphaLM API from the CyphaLM build plan.

**Data:** WikiText-2, Gutenberg texts, Python stdlib source (already on disk in WSL).

```python
# domains/d17_cyphalm_integration.py
# NOTE: This domain requires cypha_lm package to be installed.
# Import from the CyphaLM build plan repo.
from cypha_lm import CyphaLM
from cypha_lm.config import CyphaLMConfig

config = CyphaLMConfig(
    vocab_size=128,       # Character-level
    d_embed=64,
    d_state=128,
    tau_fast=1.0,
    tau_slow=20.0,
    ssm_layers=2,
    field_dim=160,
    max_experts=128,
    alpha_init=0.5,
    context_length=256,
)
model = CyphaLM(config)
```

**Experiments:**

**Experiment 17A — Bits-Per-Character on WikiText-2:**
Standard LM benchmark. Compare against:
- Bigram model
- Character-level LSTM (if available)
- CyphaDIF alone (no SSM context — ablation)

**Experiment 17B — Alpha Spectrum Emergence:**
Track α per expert and α of GRIA projection every 1000 steps. Confirm convergence to α≈0.5 distribution as per the Grand Unified Law.
Record: fraction of experts with |α−0.5| < 0.1 at each checkpoint.

**Experiment 17C — Uncertainty-Gated Generation:**
Generate 1000-character sequences from 5 prompts. At each step where epistemic_var > threshold, mark the position as "uncertain." Compute:
- What fraction of generation is uncertain vs confident?
- Are uncertain positions more syntactically complex (longer words, rare characters)?

**Experiment 17D — Online Adaptation Across Authors:**
Train on Moby Dick. Evaluate BPC on Sherlock Holmes (different vocabulary, style). Then adapt online to Sherlock Holmes for 5000 steps, re-evaluate. Measure BPC improvement and expert creation rate.

**Experiment 17E — Compression Profile:**
At each training checkpoint, compute `model.compression_profile()`:
- Lossy fraction (how much is in the distributional prior)
- Lossless fraction (epistemic residual mass)
Plot both over training. Hypothesis: lossy fraction increases monotonically as the model absorbs the corpus.

**Figure:**
- `fig17a_cyphalm_bpc_comparison.png`
- `fig17b_alpha_spectrum_emergence.png` — α histogram at steps 0, 10k, 50k, 200k
- `fig17c_uncertainty_generation_positions.png`
- `fig17d_compression_profile_over_training.png`

---

## 22. Cross-Domain Measurements

These analyses run across all completed domain experiments and produce global characterisations of CyphaDIF's properties.

### 22.1 Uncertainty Calibration (All Domains)

```python
# cross_domain/uncertainty_calibration.py
"""
For every domain where ground truth labels are available:
  1. Compute Expected Calibration Error (ECE) for classification tasks
  2. Compute predictive interval coverage for regression tasks
  3. Compute AUROC for OOD detection across domain pairs

Produces a single calibration summary table across all 17 domains.
"""
```

Expected output: a table where each row is a domain/experiment and columns are ECE, coverage@95%, OOD-AUROC.

### 22.2 Online Adaptation Speed (All Domains)

```python
# cross_domain/online_adaptation.py
"""
For each domain with a concept drift or distribution shift experiment:
  Define T_adapt = number of steps for accuracy to recover to 90% of pre-drift level.
  Compare T_adapt for CyphaDIF vs SGD baseline.

Produces a bar chart: adaptation speed across domains.
"""
```

### 22.3 Forgetting Resistance (All Domains)

```python
# cross_domain/forgetting_resistance.py
"""
Forgetting score = (Accuracy_before_drift - Accuracy_after_drift_with_recovery)
                   / Accuracy_before_drift

Lower = better (less forgetting).
CyphaDIF should have lower forgetting score than SGD across all domains.
"""
```

### 22.4 Alpha Spectrum Global (All Domains)

```python
# cross_domain/alpha_spectrum_global.py
"""
For each domain/experiment, record mean_alpha at model convergence.
Also record fraction_edge_of_chaos = fraction of experts with |α−0.5| < 0.1.

Hypothesis: across all domains, mean_alpha converges to ≈0.5 ± 0.15 at convergence.
This is the empirical validation of the Grand Unified Law.

Produce: box plot of α distributions across all 17 domains.
"""
```

---

## 23. Report Generator

```python
# report/generate_report.py
"""
Assembles all figures and metric tables into a structured PDF report.
Uses reportlab for PDF generation — no LaTeX required.

Report structure:
  1. Executive Summary (auto-generated from metrics)
  2. Per-domain results (17 sections, one per domain)
  3. Cross-domain comparisons (calibration, adaptation, forgetting, α)
  4. Appendix: full metrics table (all 60+ experiments)

Usage:
  python report/generate_report.py --output CyphaDIF_Report.pdf
"""

import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Image, Table,
                                  TableStyle, PageBreak, Spacer)
from reportlab.lib.styles import getSampleStyleSheet

def build_report(figures_dir='report/figures', tables_dir='report/tables',
                 output_path='CyphaDIF_Report.pdf'):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Title page
    story.append(Paragraph("CyphaDIF Comprehensive Test Bench Report", styles['Title']))
    story.append(Paragraph("Author: Odin Loch", styles['Normal']))
    story.append(PageBreak())

    # Load all figures in order and add to report
    for fig_file in sorted(os.listdir(figures_dir)):
        if fig_file.endswith('.png'):
            domain = fig_file.split('_')[0]  # e.g. 'fig01a'
            story.append(Paragraph(fig_file.replace('.png','').replace('_',' '), styles['Heading3']))
            story.append(Image(os.path.join(figures_dir, fig_file), width=400, height=280))
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Report written to {output_path}")

if __name__ == '__main__':
    build_report()
```

---

## 24. Master Runner

```python
# run_all.py
"""
Runs all 17 domain experiments in sequence.
Saves all metrics and figures.
Generates the final report.

Usage (WSL or Windows):
  python run_all.py                   # Run all domains
  python run_all.py --domain 5        # Run only domain 5 (chess)
  python run_all.py --from-domain 8   # Resume from domain 8
  python run_all.py --report-only     # Skip experiments, just build report

Estimated runtime on RTX 3090 / 64-core Xeon:
  Domains 01-04:  ~10 minutes
  Domains 05-07:  ~15 minutes (chess PGN parsing)
  Domains 08-09:  ~20 minutes (MNIST + 20News)
  Domains 10-12:  ~15 minutes
  Domains 13-16:  ~20 minutes
  Domain 17:      ~60 minutes (CyphaLM — skip if not yet implemented)
  Cross-domain:   ~10 minutes
  Report:         ~2 minutes
  Total:          ~2.5 hours (without domain 17)
"""

import argparse, importlib, sys, os, time

DOMAINS = [
    ('d01', 'domains.d01_statistical_baselines', 'run'),
    ('d02', 'domains.d02_regression',            'run'),
    ('d03', 'domains.d03_classification',        'run'),
    ('d04', 'domains.d04_generation_language',   'run'),
    ('d05', 'domains.d05_chess',                 'run'),
    ('d06', 'domains.d06_go',                    'run'),
    ('d07', 'domains.d07_poker',                 'run'),
    ('d08', 'domains.d08_computer_vision',       'run'),
    ('d09', 'domains.d09_documents',             'run'),
    ('d10', 'domains.d10_time_series',           'run'),
    ('d11', 'domains.d11_reinforcement_learning','run'),
    ('d12', 'domains.d12_anomaly_detection',     'run'),
    ('d13', 'domains.d13_compression',           'run'),
    ('d14', 'domains.d14_symbolic_regression',   'run'),
    ('d15', 'domains.d15_adversarial_robustness','run'),
    ('d16', 'domains.d16_multitask',             'run'),
    ('d17', 'domains.d17_cyphalm_integration',   'run'),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',      type=int, default=None)
    parser.add_argument('--from-domain', type=int, default=1)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()

    if args.report_only:
        from report.generate_report import build_report
        build_report()
        return

    domains_to_run = DOMAINS
    if args.domain:
        domains_to_run = [d for d in DOMAINS if int(d[0][1:]) == args.domain]
    elif args.from_domain > 1:
        domains_to_run = [d for d in DOMAINS if int(d[0][1:]) >= args.from_domain]

    for tag, module_path, fn_name in domains_to_run:
        print(f"\n{'='*60}")
        print(f"Running {tag}: {module_path}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            mod = importlib.import_module(module_path)
            getattr(mod, fn_name)()
            print(f"  Completed in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Cross-domain analyses
    print("\nRunning cross-domain analyses...")
    from cross_domain import uncertainty_calibration, online_adaptation, \
                              forgetting_resistance, alpha_spectrum_global
    uncertainty_calibration.run()
    online_adaptation.run()
    forgetting_resistance.run()
    alpha_spectrum_global.run()

    # Build report
    from report.generate_report import build_report
    build_report()
    print("\nDone. Report: CyphaDIF_Report.pdf")

if __name__ == '__main__':
    main()
```

---

## Appendix A — CyphaDIF-Specific Measurements Reference

Every experiment should record these measurements in addition to standard accuracy/RMSE. These are what makes the report valuable beyond a standard ML benchmark.

| Measurement | What it means | Where it matters most |
|---|---|---|
| `mean_epistemic_var` | Average parameter uncertainty across test set | OOD detection, calibration, RL exploration |
| `mean_aleatoric_var` | Average irreducible noise across test set | Noisy labels, stochastic tasks, financial data |
| `epistemic_vs_error_spearman` | Rank correlation of uncertainty with actual error | Calibration — should be > 0.3 everywhere |
| `expert_count` | Number of experts at convergence | Model complexity, task diversity |
| `expert_count_growth_curve` | Expert count vs training step | Compression — count should plateau |
| `mean_alpha` | Mean GRIA α across all active experts | Grand Unified Law validation |
| `fraction_edge_of_chaos` | Fraction of experts with α ∈ [0.4, 0.6] | Self-organisation quality |
| `ood_epistemic_ratio` | Epistemic_var on OOD / epistemic_var on in-dist | OOD detection signal-to-noise |
| `adaptation_steps_to_90pct` | Steps to recover 90% accuracy after drift | Online learning speed |
| `forgetting_score` | Accuracy drop on Task A after training Task B | Catastrophic forgetting |
| `coverage_at_95pct` | NIG predictive interval coverage | Regression calibration |

---

## Appendix B — Failure Mode Reference

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| Expert count never grows | Novelty threshold too high | Check epistemic_var on clearly OOD inputs — should be very high | Lower novelty threshold |
| Expert count explodes | Novelty threshold too low | Monitor count every 100 steps | Raise threshold or cap max_experts |
| Epistemic_var same on in-dist and OOD | DIF field_dim too small to distinguish | Run 22.4 — α should be near 0 if no structure found | Increase field_dim to 256+ |
| Aleatoric_var ≈ 0 always | NIG β₀ too small (prior too tight) | Inject pure noise — aleatoric should be non-zero | Increase β₀ |
| Accuracy lower than dummy classifier | Feature encoding is degenerate (all zeros or all same) | Check encoder output with assert np.std(X) > 0.01 | Fix encoder normalisation |
| Classification stuck at uniform distribution | Labels not being passed correctly (all same label) | Check label diversity in train loop | Fix label encoding |
| α collapses to 0 on all tasks | NIG updates too aggressive (lr too high) | Monitor μ_n divergence | Reduce effective update rate |
| α stuck at 1 on all tasks | Features all zero — no structure to find | Add noise to verify model can learn anything | Fix input encoding |
| OOD AUROC < 0.6 | Model routes OOD to existing experts confidently | Check if OOD data is actually different — compute feature distance | Increase field_dim, lower novelty threshold |
| NaN after many steps | Float32 accumulation in NIG statistics | Check β_n for divergence | Add periodic NIG parameter clipping: β_n = min(β_n, 1e6) |

---

## Appendix C — Quick Reference: Dataset Sizes and Acquisition Time

| Dataset | Size | Acquisition | Notes |
|---|---|---|---|
| sklearn built-ins | 0 MB | Instant | iris, wine, breast_cancer, digits, california_housing, diabetes |
| MNIST | 11 MB | wget (~30s) | Train: 60k, Test: 10k, 28×28 greyscale |
| WikiText-2 | 4 MB | wget (~10s) | 2M train tokens, standard LM benchmark |
| 20 Newsgroups | 14 MB | sklearn auto (~60s) | 18,846 posts, 20 categories |
| NSL-KDD | 7 MB | wget (~15s) | 148,517 train / 22,544 test connections |
| ECG5000 | 3 MB | wget (~10s) | 500 train / 4500 test, 140 timesteps, 5 classes |
| Canterbury Corpus | 2 MB | wget (~5s) | 11 reference files for compression benchmarking |
| Gutenberg (3 books) | 2 MB | wget (~10s) | Moby Dick, Sherlock Holmes, Alice |
| Kasparov PGN | ~2 MB | wget (~5s) | ~2000 grandmaster games |
| Financial (yfinance) | <1 MB | API call (~5s) | SPY, QQQ, GLD, TLT, 5 years daily |
| UCI regression datasets | <1 MB each | ucimlrepo (~5s each) | Concrete, Energy, Auto MPG, Wine Quality |
| Go positions | 0 MB | Synthetic | Generated in-script with sgfmill |
| Poker hands | 0 MB | Synthetic | Generated in-script with treys |
| RL environments | 0 MB | Synthetic | CartPole, GridWorld in pure numpy |
| Feynman equations | 0 MB | Synthetic | Generated from known equations |
| **Total** | **~46 MB** | **~3 minutes** | |

---

*End of CyphaDIF Test Bench Specification. Version 1.0.*  
*All 17 domains, 60+ named experiments, 40+ figures, 1 consolidated PDF report.*
