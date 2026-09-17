#!/usr/bin/env python3
"""
game_benchmark_v4.py — Cypha HRNA: learn and play Chess, Poker, and Go
═══════════════════════════════════════════════════════════════════════════════

All three games follow the same pipeline:
  TRAIN  →  play complete games, extract labelled (features, decision) pairs
  TEST   →  Cypha plays against a real opponent, report results

Chess
  Train : Stockfish self-play at depth 18; every position labelled best_move /
          other_move.  Cypha does behavioural cloning.
  Test  : Cypha (White) vs Stockfish depth 20, Skill Level 20 (hardest).
  Score : Wins / Draws / Losses

Poker  (heads-up Texas Hold'em, no-limit)
  Train : Deal N complete hands to showdown; at each street the GTO-optimal
          action (based on real equity computed by treys Monte Carlo) is
          labelled "best_action"; random alternatives labelled "other_action".
  Test  : Cypha plays heads-up vs a calibrated rule-based opponent for
          N_TEST_HANDS hands; initial stack 1000 chips each.
  Score : Net chips won / lost, hands won percentage

Go  (9×9, Chinese rules, pure Python — no external library needed)
  Train : Self-play games between two greedy bots; Cypha watches and labels
          every move the winning side played as "best_move" and losing side
          moves as "other_move" (credit assignment by game outcome).
  Test  : Cypha (Black) vs a greedy territory bot (White) for N_TEST_GAMES.
  Score : Wins / Draws / Losses by final territory count

Just run:
    python game_benchmark_v4.py

Requirements:
    pip install python-chess treys
    Stockfish binary on PATH  (or set STOCKFISH_PATH env var)
    Cypha.py in the same directory
"""

import sys, os, time, random, tempfile, shutil, collections, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Cypha import CyphaStateful

# ── Optional engine imports ───────────────────────────────────────────────────
try:
    import chess, chess.engine
    CHESS_OK = True
except ImportError:
    CHESS_OK = False
    print("WARNING: python-chess not installed (pip install python-chess). "
          "Chess domain will be skipped.")

try:
    from treys import Evaluator, Card, Deck
    POKER_OK = True
    _EVALUATOR = Evaluator()
except ImportError:
    POKER_OK = False
    print("WARNING: treys not installed (pip install treys). "
          "Poker domain will be skipped.")

GO_OK = True   # pure Python, always available

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

STOCKFISH_PATH   = os.environ.get("STOCKFISH_PATH", "stockfish")

# Chess
N_CHESS_TRAIN    = 1216    # complete Stockfish self-play games  (~12.7h of 16h budget)
N_CHESS_TEST     = 20      # games Cypha plays vs Stockfish
CHESS_TRAIN_DEPTH= 18
CHESS_TEST_DEPTH = 20
CHESS_SKILL      = 20      # 0-20, 20 = hardest
CHESS_NEG_SAMP   = 1       # "other_move" per position — 1:1 ratio improves best_recall
CHESS_MAX_MOVES  = 200

# Poker
N_POKER_TRAIN    = 30000   # complete hands dealt for training   (~0.3h)
N_POKER_TEST     = 200     # hands Cypha plays vs rule-based opponent
POKER_MC_ITERS   = 300     # Monte Carlo equity rollouts per decision
STARTING_STACK   = 1000    # chips per player
BIG_BLIND        = 10

# Go
N_GO_TRAIN       = 3000    # complete 9×9 self-play games        (~3.0h)
N_GO_TEST        = 30      # games Cypha plays vs greedy bot
GO_SIZE          = 9
GO_KOMI          = 6.5     # komi for White (Chinese rules)
GO_MAX_MOVES     = 200

RNG = random.Random(42)
NP_RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
#  CYPHA INSTANCES  (one per domain so they don't share weights)
# ─────────────────────────────────────────────────────────────────────────────

def make_cypha(tmp_dir, name):
    ckpt = os.path.join(tmp_dir, f"ckpt_{name}")
    return CyphaStateful(feature_dim=512, resonance_dim=256,
                         checkpoint_root=ckpt)


# ══════════════════════════════════════════════════════════════════════════════
#  ██████  ██   ██ ███████ ███████ ███████
#  ██      ██   ██ ██      ██      ██
#  ██      ███████ █████   ███████ ███████
#  ██      ██   ██ ██          ██      ██
#  ██████  ██   ██ ███████ ███████ ███████
# ══════════════════════════════════════════════════════════════════════════════

_PIECE_NAMES = {
    chess.PAWN:"pawn", chess.KNIGHT:"knight", chess.BISHOP:"bishop",
    chess.ROOK:"rook", chess.QUEEN:"queen", chess.KING:"king",
} if CHESS_OK else {}

_PIECE_VALS = {
    chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330,
    chess.ROOK:500, chess.QUEEN:900, chess.KING:0,
} if CHESS_OK else {}

_CHESS_OPENINGS = [
    ["e2e4","e7e5"],  ["e2e4","c7c5"],  ["d2d4","d7d5"],
    ["d2d4","g8f6"],  ["e2e4","e7e6"],  ["g1f3","d7d5"],
    ["e2e4","c7c6"],  ["c2c4","e7e5"],  ["e2e4","d7d5"],
    ["d2d4","f7f5"],  ["b1c3","d7d5"],  ["e2e4","g7g6"],
    ["d2d4","e7e6"],  ["e2e4","b8c6"],  ["g2g3","d7d5"],
    ["f2f4","d7d5"],  ["b2b4","e7e5"],  ["d2d4","d7d6"],
    ["e2e4","a7a6"],  ["c2c4","c7c5"],
]


def _open_stockfish():
    try:
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        eng.configure({"Skill Level": CHESS_SKILL})
        return eng
    except FileNotFoundError:
        print(f"  ERROR: Stockfish not found at '{STOCKFISH_PATH}'")
        print("  Download: https://stockfishchess.org/download/")
        print("  Then:  export STOCKFISH_PATH=/path/to/stockfish")
        return None


def chess_position_features(board, move, eval_cp=0):
    """Build feature token string for a board + candidate move."""
    # Material
    wm = sum(_PIECE_VALS.get(pt,0)*len(board.pieces(pt,chess.WHITE)) for pt in _PIECE_VALS)
    bm = sum(_PIECE_VALS.get(pt,0)*len(board.pieces(pt,chess.BLACK)) for pt in _PIECE_VALS)
    mat_diff = wm - bm

    # Mobility
    orig = board.turn
    board.turn = chess.WHITE; mob_w = board.legal_moves.count()
    board.turn = chess.BLACK; mob_b = board.legal_moves.count()
    board.turn = orig

    total_pieces = len(board.piece_map())

    # Pawn structure
    wp = list(board.pieces(chess.PAWN, chess.WHITE))
    bp = list(board.pieces(chess.PAWN, chess.BLACK))
    wf = [chess.square_file(s) for s in wp]
    bf = [chess.square_file(s) for s in bp]
    iso_w = sum(1 for f in set(wf) if (f-1) not in wf and (f+1) not in wf)
    iso_b = sum(1 for f in set(bf) if (f-1) not in bf and (f+1) not in bf)
    pass_w = sum(1 for s in wp
                 if not any(chess.square_file(o) in
                             (chess.square_file(s)-1, chess.square_file(s),
                              chess.square_file(s)+1)
                             and chess.square_rank(o) > chess.square_rank(s)
                             for o in bp))

    # Phase
    if board.fullmove_number <= 12 and total_pieces >= 26: phase = "opening"
    elif total_pieces >= 16:   phase = "middlegame"
    elif total_pieces >= 8:    phase = "endgame_early"
    else:                       phase = "endgame_late"

    # Move details
    piece   = board.piece_at(move.from_square)
    pname   = _PIECE_NAMES.get(piece.piece_type, "unknown") if piece else "unknown"
    to_f    = chess.square_file(move.to_square)
    to_r    = chess.square_rank(move.to_square)
    is_cap  = board.is_capture(move)
    is_cast = board.is_castling(move)
    board.push(move); gives_chk = board.is_check(); board.pop()

    region = "qs" if to_f <= 2 else ("ctr" if to_f <= 5 else "ks")
    if eval_cp > 300:   eb = "winning"
    elif eval_cp > 80:  eb = "better"
    elif eval_cp > -80: eb = "equal"
    elif eval_cp > -300:eb = "worse"
    else:               eb = "losing"

    wk = board.king(chess.WHITE); bk = board.king(chess.BLACK)
    ks_w = len(board.attackers(chess.BLACK, wk)) if wk else 0
    ks_b = len(board.attackers(chess.WHITE, bk)) if bk else 0

    toks = [
        f"phase_{phase}", f"move_{board.fullmove_number}",
        f"pieces_{total_pieces}", f"mat_{mat_diff:+d}", f"eval_{eb}",
        f"mob_{mob_w-mob_b:+d}", f"ks_w_{ks_w}", f"ks_b_{ks_b}",
        f"iso_{iso_w+iso_b}", f"pass_{pass_w}",
        f"piece_{pname}", f"to_{region}_r{to_r}",
        f"cap_{'y' if is_cap else 'n'}",
        f"chk_{'y' if gives_chk else 'n'}",
        f"cast_{'y' if is_cast else 'n'}",
    ]
    if gives_chk:    toks.append("gives_check")
    if is_cap:
        cap = board.piece_at(move.to_square)
        if cap: toks.append(f"caps_{_PIECE_NAMES.get(cap.piece_type,'x')}")
    if mob_w - mob_b > 8:  toks.append("mobility_advantage")
    if ks_w >= 2:           toks.append("king_exposed")
    if total_pieces <= 10:  toks.append("endgame_kings")
    if pass_w >= 2:         toks.append("passed_pawns")
    return " ".join(toks)


def chess_collect_training(n_games):
    eng = _open_stockfish()
    if eng is None:
        return []
    pairs = []; t0 = time.time()
    print(f"  Playing {n_games} training games (depth {CHESS_TRAIN_DEPTH})...")
    for gi in range(n_games):
        board = chess.Board()
        seed  = RNG.choice(_CHESS_OPENINGS)
        for uci in seed:
            try: board.push_uci(uci)
            except: break

        for _ in range(CHESS_MAX_MOVES):
            if board.is_game_over(): break
            info  = eng.analyse(board, chess.engine.Limit(depth=CHESS_TRAIN_DEPTH))
            score = info["score"].white()
            ecp   = int(score.score(mate_score=30000) or 0) if not score.is_mate() else (
                    30000 if (score.mate() or 0) > 0 else -30000)
            pv    = info.get("pv", [])
            best  = pv[0] if pv and pv[0] in board.legal_moves else None
            if best is None:
                legal = list(board.legal_moves)
                if not legal: break
                best = RNG.choice(legal)

            pairs.append((chess_position_features(board, best, ecp), "best_move"))
            others = [m for m in board.legal_moves if m != best]
            for m in RNG.sample(others, min(CHESS_NEG_SAMP, len(others))):
                pairs.append((chess_position_features(board, m, ecp), "other_move"))

            board.push(best)

        elapsed = time.time()-t0
        print(f"    game {gi+1:>3}/{n_games}  pairs={len(pairs):>6,}  "
              f"{elapsed:.0f}s  {(gi+1)/elapsed:.1f}g/s", end="\r")

    eng.quit(); print()
    print(f"  Chess training data: {len(pairs):,} examples from {n_games} games "
          f"({time.time()-t0:.1f}s)")
    return pairs


def chess_pick_move(cypha, board, eval_cp=0):
    legal = list(board.legal_moves)
    if not legal: return None
    if len(legal) == 1: return legal[0]
    best_move = None; best_conf = -1.0
    for m in legal:
        feat = chess_position_features(board, m, eval_cp)
        try:
            lbl, conf = cypha.infer(feat, verbose=False)
            if lbl == "best_move" and conf > best_conf:
                best_conf = conf; best_move = m
        except: pass
    return best_move or RNG.choice(legal)


def chess_run_test(cypha):
    eng = _open_stockfish()
    if eng is None:
        return 0, 0, 0, []
    wins = draws = losses = 0; results = []; t0 = time.time()
    print(f"  Playing {N_CHESS_TEST} test games vs Stockfish "
          f"(depth {CHESS_TEST_DEPTH}, skill {CHESS_SKILL}/20)...")

    for i in range(N_CHESS_TEST):
        board = chess.Board()
        seed  = RNG.choice(_CHESS_OPENINGS)
        for uci in seed:
            try: board.push_uci(uci)
            except: break

        for _ in range(CHESS_MAX_MOVES):
            if board.is_game_over(): break
            if board.turn == chess.WHITE:
                try:
                    inf = eng.analyse(board, chess.engine.Limit(depth=6))
                    s   = inf["score"].white()
                    ecp = int(s.score(mate_score=30000) or 0) if not s.is_mate() else (
                          30000 if (s.mate() or 0) > 0 else -30000)
                except: ecp = 0
                move = chess_pick_move(cypha, board, ecp)
            else:
                r    = eng.play(board, chess.engine.Limit(depth=CHESS_TEST_DEPTH))
                move = r.move
            if move is None or move not in board.legal_moves:
                legal = list(board.legal_moves)
                if not legal: break
                move = RNG.choice(legal)
            board.push(move)

        if board.is_game_over():
            out = board.outcome()
            if out and out.winner == chess.WHITE:   r="1-0";   wins+=1
            elif out and out.winner == chess.BLACK: r="0-1";   losses+=1
            else:                                    r="1/2";   draws+=1
        else:
            r="1/2"; draws+=1
        results.append(r)
        marker = {"1-0":"WIN","0-1":"LOSS","1/2":"DRAW"}[r]
        print(f"    game {i+1:>3}/{N_CHESS_TEST}  {r}  "
              f"[W={wins} D={draws} L={losses}]  {time.time()-t0:.0f}s")

    eng.quit()
    return wins, draws, losses, results


# ══════════════════════════════════════════════════════════════════════════════
#  ██████   ██████  ██   ██ ███████ ██████
#  ██   ██ ██    ██ ██  ██  ██      ██   ██
#  ██████  ██    ██ █████   █████   ██████
#  ██      ██    ██ ██  ██  ██      ██   ██
#  ██       ██████  ██   ██ ███████ ██   ██
# ══════════════════════════════════════════════════════════════════════════════

_SUITS  = ['s','h','d','c']
_RANKS  = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
_RANK_V = {r:i+2 for i,r in enumerate(_RANKS)}

def _new_deck():
    return [r+s for r in _RANKS for s in _SUITS]

def _deal(deck, n):
    cards = RNG.sample(deck, n)
    for c in cards: deck.remove(c)
    return cards

def _treys_card(c):
    rank = c[0].upper()
    suit = c[1].lower()
    if rank == 'T': rank = 'T'
    return Card.new(rank + suit)

def _hand_equity_mc(hole, board_cards, n_iter=POKER_MC_ITERS):
    """Monte Carlo equity using treys evaluator."""
    deck = _new_deck()
    for c in hole + board_cards:
        if c in deck: deck.remove(c)
    wins = ties = 0
    for _ in range(n_iter):
        remaining = deck[:]
        RNG.shuffle(remaining)
        vill   = remaining[:2]
        needed = 5 - len(board_cards)
        runout = remaining[2:2+needed]
        full   = board_cards + runout
        try:
            tc_h = [_treys_card(c) for c in hole]
            tc_v = [_treys_card(c) for c in vill]
            tc_b = [_treys_card(c) for c in full]
            hs = _EVALUATOR.evaluate(tc_b, tc_h)
            vs = _EVALUATOR.evaluate(tc_b, tc_v)
            if hs < vs:   wins += 1
            elif hs == vs: ties += 1
        except: pass
    return round(100.0*(wins + 0.5*ties)/n_iter, 1)


def _gto_action(equity, pot_odds_pct, spr, street):
    """GTO-approximate action from real equity."""
    mdf = 1.0 - pot_odds_pct/100.0
    if equity >= 65:                                   return "raise"
    if equity >= 50 and pot_odds_pct <= 33:            return "raise"
    if equity >= 38 and pot_odds_pct <= 25:            return "call"
    if equity >= 28 and spr > 4 and street == "flop":  return "call"   # draws
    if equity >= 20 and pot_odds_pct <= 15:            return "call"
    return "fold"


_HAND_BUCKETS = [
    (1, "royal_flush"), (10, "straight_flush"), (166, "four_of_a_kind"),
    (322, "full_house"), (1599, "flush"), (1609, "straight"),
    (2467, "three_of_a_kind"), (3325, "two_pair"), (6185, "pair"),
    (7462, "high_card"),
]

def _hand_bucket(rank):
    for threshold, name in _HAND_BUCKETS:
        if rank <= threshold: return name
    return "high_card"


def poker_features(hole, board_cards, equity, pot, stack_hero, stack_vill,
                   street, position, action):
    """Build feature token string for a poker decision."""
    pot_odds  = round(100.0*BIG_BLIND/(pot + BIG_BLIND), 1) if pot > 0 else 50.0
    spr       = round(min(stack_hero, stack_vill) / max(pot, 1), 1)
    fold_eq   = round(max(0.0, (100.0-equity)*0.60), 1)
    is_suited = hole[0][1] == hole[1][1] if len(hole) >= 2 else False
    r1 = _RANK_V.get(hole[0][0], 2) if hole else 2
    r2 = _RANK_V.get(hole[1][0], 2) if len(hole) > 1 else 2
    high_card = max(r1, r2); low_card = min(r1, r2)
    connected = abs(r1-r2) <= 2

    if len(board_cards) >= 3:
        try:
            tc_h = [_treys_card(c) for c in hole]
            tc_b = [_treys_card(c) for c in board_cards[:3]]
            rank = _EVALUATOR.evaluate(tc_b, tc_h)
            made = _hand_bucket(rank)
        except: made = "unknown"
    else:
        made = "preflop"

    if   equity >= 70: eq_b = "strong"
    elif equity >= 50: eq_b = "ahead"
    elif equity >= 35: eq_b = "marginal"
    elif equity >= 20: eq_b = "behind"
    else:               eq_b = "weak"

    toks = [
        f"street_{street}", f"pos_{position}",
        f"eq_{eq_b}", f"equity_{equity:.0f}",
        f"pot_{min(pot,2000)}", f"spr_{spr:.1f}",
        f"pot_odds_{pot_odds:.0f}", f"fold_eq_{fold_eq:.0f}",
        f"high_{high_card}", f"low_{low_card}",
        f"suited_{'y' if is_suited else 'n'}",
        f"connected_{'y' if connected else 'n'}",
        f"made_{made}",
        f"stack_h_{min(stack_hero,3000)}",
        f"board_cards_{len(board_cards)}",
    ]
    if equity >= 60:        toks.append("value_territory")
    if equity < 30 and fold_eq > 40: toks.append("bluff_territory")
    if spr < 3:             toks.append("short_stack")
    if connected:           toks.append("connected_hole")
    if is_suited:           toks.append("suited_hole")
    return " ".join(toks)


def poker_collect_training(n_hands):
    if not POKER_OK:
        return []
    pairs = []; t0 = time.time()
    print(f"  Dealing {n_hands} training hands (MC iters={POKER_MC_ITERS})...")

    streets = ["preflop","flop","turn","river"]

    for hi in range(n_hands):
        deck = _new_deck()
        hero = _deal(deck, 2); vill = _deal(deck, 2)
        flop = _deal(deck, 3); turn = _deal(deck, 1); river = _deal(deck, 1)
        pot  = BIG_BLIND * 2
        stk  = STARTING_STACK
        pos  = RNG.choice(["btn","bb"])

        board_progress = [[], flop, flop+turn, flop+turn+river]

        for si, street in enumerate(streets):
            bc     = board_progress[si]
            equity = _hand_equity_mc(hero, bc)
            bet    = round(pot * 0.5)
            pot_odds = round(100.0*bet/(pot+bet), 1)
            spr    = round(stk/max(pot,1), 1)

            best_action = _gto_action(equity, pot_odds, spr, street)
            pairs.append((poker_features(hero, bc, equity, pot, stk, stk,
                                         street, pos, best_action),
                          "best_action"))

            # Alternative (non-optimal) actions — 1 negative per decision keeps ratio balanced
            all_actions = [a for a in ("fold","call","raise") if a != best_action]
            for alt in RNG.sample(all_actions, min(1, len(all_actions))):
                pairs.append((poker_features(hero, bc, equity, pot, stk, stk,
                                             street, pos, alt),
                              "other_action"))

            # Update pot for next street
            if best_action == "fold": break
            elif best_action == "raise": pot += bet*2
            else: pot += bet

        if (hi+1) % 500 == 0:
            elapsed = time.time()-t0
            print(f"    hand {hi+1:>5}/{n_hands}  pairs={len(pairs):>6,}  "
                  f"{elapsed:.0f}s  {(hi+1)/elapsed:.0f}h/s", end="\r")

    print()
    print(f"  Poker training data: {len(pairs):,} examples from {n_hands} hands "
          f"({time.time()-t0:.1f}s)")
    return pairs


def _rule_based_action(equity, pot, stack, street):
    """Simple rule-based opponent for testing."""
    pot_odds = 100.0*BIG_BLIND/(pot+BIG_BLIND) if pot > 0 else 50.0
    if equity >= 60: return "raise"
    if equity >= 40: return "call"
    if pot_odds < 15 and equity >= 25: return "call"
    return "fold"


def poker_pick_action(cypha, hole, board_cards, equity, pot, stack_h, stack_v,
                      street, position):
    actions = ["fold","call","raise"]
    best_action = None; best_conf = -1.0
    for a in actions:
        feat = poker_features(hole, board_cards, equity, pot, stack_h, stack_v,
                              street, position, a)
        try:
            lbl, conf = cypha.infer(feat, verbose=False)
            if lbl == "best_action" and conf > best_conf:
                best_conf = conf; best_action = a
        except: pass
    return best_action or "call"


def poker_run_test(cypha):
    if not POKER_OK:
        return 0, 0, 0
    wins = losses = draws = 0
    hero_net = 0.0
    t0 = time.time()
    print(f"  Playing {N_POKER_TEST} test hands (Cypha vs rule-based opponent)...")

    for hi in range(N_POKER_TEST):
        deck      = _new_deck()
        hero      = _deal(deck, 2)
        vill      = _deal(deck, 2)
        flop      = _deal(deck, 3)
        turn      = _deal(deck, 1)
        river     = _deal(deck, 1)
        pot       = BIG_BLIND * 2
        stk_h     = STARTING_STACK
        stk_v     = STARTING_STACK
        hero_bet  = 0.0
        vill_bet  = 0.0
        position  = RNG.choice(["btn","bb"])
        folded    = False
        streets   = ["preflop","flop","turn","river"]
        bp_map    = {
            "preflop": [],
            "flop":    flop,
            "turn":    flop+turn,
            "river":   flop+turn+river,
        }

        for street in streets:
            bc     = bp_map[street]
            equity = _hand_equity_mc(hero, bc, n_iter=200)
            bet    = round(pot * 0.5)

            # Cypha acts first
            h_action = poker_pick_action(cypha, hero, bc, equity,
                                         pot, stk_h, stk_v, street, position)
            if h_action == "fold":
                hero_net -= hero_bet; folded = True; break
            elif h_action == "raise":
                contributed = min(bet, stk_h)
                hero_bet  += contributed
                pot       += contributed
                stk_h     -= contributed

            # Villain responds
            v_equity = 100.0 - equity
            v_action = _rule_based_action(v_equity, pot, stk_v, street)
            if v_action == "fold":
                hero_net += vill_bet; break
            elif v_action == "raise":
                contributed = min(bet, stk_v)
                vill_bet  += contributed
                pot       += contributed
                stk_v     -= contributed
            else:
                vill_bet  += min(bet//2, stk_v)
                pot       += min(bet//2, stk_v)
                stk_v     -= min(bet//2, stk_v)

        if not folded:
            # Showdown
            full_board = flop + turn + river
            try:
                tc_h = [_treys_card(c) for c in hero]
                tc_v = [_treys_card(c) for c in vill]
                tc_b = [_treys_card(c) for c in full_board]
                hs = _EVALUATOR.evaluate(tc_b, tc_h)
                vs = _EVALUATOR.evaluate(tc_b, tc_v)
                if hs < vs:    wins   += 1; hero_net += pot * 0.5
                elif hs > vs:  losses += 1; hero_net -= hero_bet
                else:          draws  += 1
            except:
                draws += 1

        if (hi+1) % 50 == 0:
            print(f"    hand {hi+1:>4}/{N_POKER_TEST}  "
                  f"W={wins} D={draws} L={losses}  "
                  f"net={hero_net:+.0f} chips  {time.time()-t0:.0f}s", end="\r")

    print()
    return wins, draws, losses, hero_net


# ══════════════════════════════════════════════════════════════════════════════
#   ██████   ██████
#  ██       ██    ██
#  ██   ███ ██    ██
#  ██    ██ ██    ██
#   ██████   ██████
# ══════════════════════════════════════════════════════════════════════════════

# Pure Python 9×9 Go engine
# Board: numpy int8  0=empty  1=Black  -1=White

class GoBoard:
    SIZE = GO_SIZE

    def __init__(self):
        self.board      = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        self.ko_point   = None   # (row, col) forbidden by ko
        self.captures   = {1: 0, -1: 0}   # stones captured
        self.history    = set()  # board hash history for superko

    def copy(self):
        g = GoBoard()
        g.board    = self.board.copy()
        g.ko_point = self.ko_point
        g.captures = dict(self.captures)
        g.history  = set(self.history)
        return g

    def _hash(self):
        return self.board.tobytes()

    def _liberties(self, r, c):
        color = self.board[r, c]
        if color == 0: return set(), set()
        S = self.SIZE
        visited = set(); queue = [(r,c)]; visited.add((r,c))
        group = []; libs = set()
        while queue:
            rr, cc = queue.pop(); group.append((rr,cc))
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr,nc = rr+dr,cc+dc
                if 0 <= nr < S and 0 <= nc < S:
                    if self.board[nr,nc] == color and (nr,nc) not in visited:
                        visited.add((nr,nc)); queue.append((nr,nc))
                    elif self.board[nr,nc] == 0:
                        libs.add((nr,nc))
        return set(group), libs

    def is_legal(self, r, c, color):
        S = self.SIZE
        if not (0 <= r < S and 0 <= c < S): return False
        if self.board[r,c] != 0: return False
        if self.ko_point == (r, c): return False

        # Simulate placement
        test = self.board.copy()
        test[r,c] = color
        # Check captures of opponent
        opp = -color
        any_capture = False
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<S and 0<=nc<S and test[nr,nc]==opp:
                grp,libs = self._group_libs_on(test,nr,nc)
                if not libs:
                    any_capture = True
        if any_capture: return True
        # Check own liberties after placement (suicide rule — not allowed)
        grp,libs = self._group_libs_on(test,r,c)
        return len(libs) > 0

    def _group_libs_on(self, board, r, c):
        S = self.SIZE; color = board[r,c]
        if color == 0: return set(), set()
        visited = set(); queue = [(r,c)]; visited.add((r,c))
        group = []; libs = set()
        while queue:
            rr,cc = queue.pop(); group.append((rr,cc))
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr,nc=rr+dr,cc+dc
                if 0<=nr<S and 0<=nc<S:
                    if board[nr,nc]==color and (nr,nc) not in visited:
                        visited.add((nr,nc)); queue.append((nr,nc))
                    elif board[nr,nc]==0:
                        libs.add((nr,nc))
        return set(group), libs

    def place(self, r, c, color):
        """Place stone, capture, update ko. Returns captured count."""
        S = self.SIZE
        self.board[r,c] = color
        opp = -color
        captured = 0; ko_candidate = None

        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<S and 0<=nc<S and self.board[nr,nc]==opp:
                grp,libs = self._group_libs_on(self.board,nr,nc)
                if not libs:
                    captured += len(grp)
                    for (gr,gc) in grp: self.board[gr,gc]=0
                    if len(grp)==1:
                        ko_candidate=(nr,nc)

        self.captures[color] += captured
        # Ko rule: if exactly 1 stone captured and the placed stone has exactly 1 liberty
        grp,libs = self._group_libs_on(self.board,r,c)
        if captured == 1 and len(libs) == 1:
            self.ko_point = ko_candidate
        else:
            self.ko_point = None

        self.history.add(self._hash())
        return captured

    def legal_moves(self, color):
        S = self.SIZE
        return [(r,c) for r in range(S) for c in range(S)
                if self.is_legal(r,c,color)]

    def score(self):
        """Chinese scoring: territory + stones on board."""
        S = self.SIZE
        visited = np.zeros((S,S), dtype=bool)
        b_terr = w_terr = 0
        for r in range(S):
            for c in range(S):
                if self.board[r,c] != 0 or visited[r,c]: continue
                region = []; adj_colors = set(); stack = [(r,c)]
                while stack:
                    rr,cc = stack.pop()
                    if visited[rr,cc]: continue
                    visited[rr,cc]=True; region.append((rr,cc))
                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<S and 0<=nc<S:
                            if self.board[nr,nc]==0 and not visited[nr,nc]:
                                stack.append((nr,nc))
                            elif self.board[nr,nc]!=0:
                                adj_colors.add(int(self.board[nr,nc]))
                if adj_colors == {1}:    b_terr += len(region)
                elif adj_colors == {-1}: w_terr += len(region)
        b_stones = int(np.sum(self.board==1))
        w_stones = int(np.sum(self.board==-1))
        b_score  = b_terr + b_stones
        w_score  = w_terr + w_stones + GO_KOMI
        return b_score, w_score, b_terr, w_terr


def go_board_features(board, move_rc, color, move_num):
    """Feature token string for a Go position + candidate move."""
    S = GO_SIZE
    r, c = move_rc

    b_stones = int(np.sum(board.board==1))
    w_stones = int(np.sum(board.board==-1))
    total    = b_stones + w_stones

    b_sc, w_sc, b_terr, w_terr = board.score()
    score_diff = b_sc - w_sc   # positive = Black winning

    # Group stats for candidate move
    # Simulate placement to count liberties gained
    test = board.board.copy(); test[r,c] = color
    opp = -color

    # Count how many opponent groups we'd capture
    captures_made = 0
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr,nc=r+dr,c+dc
        if 0<=nr<S and 0<=nc<S and test[nr,nc]==opp:
            grp,libs = board._group_libs_on(test,nr,nc)
            if not libs: captures_made += len(grp)

    # Own group liberties after move
    grp,libs = board._group_libs_on(test,r,c)
    own_libs  = len(libs)

    # Adjacent opponent stones in atari
    opp_atari = 0
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr,nc=r+dr,c+dc
        if 0<=nr<S and 0<=nc<S and board.board[nr,nc]==opp:
            grp,libs = board._group_libs_on(board.board,nr,nc)
            if len(libs)==1: opp_atari += 1

    # Position zones
    region = "corner" if (r in (0,S-1) and c in (0,S-1)) else (
             "edge"   if (r==0 or r==S-1 or c==0 or c==S-1) else "center")

    if   move_num < 15:  phase = "opening"
    elif move_num < 60:  phase = "middle"
    elif move_num < 120: phase = "endgame"
    else:                phase = "yose"

    if   score_diff >  10: sb = "b_winning"
    elif score_diff >   2: sb = "b_ahead"
    elif score_diff >  -2: sb = "even"
    elif score_diff >  -10:sb = "w_ahead"
    else:                  sb = "w_winning"

    toks = [
        f"mv_{move_num}", f"phase_{phase}",
        f"b_stones_{b_stones}", f"w_stones_{w_stones}", f"total_{total}",
        f"score_{sb}", f"b_terr_{b_terr}", f"w_terr_{w_terr}",
        f"move_row_{r}", f"move_col_{c}", f"region_{region}",
        f"own_libs_{min(own_libs,8)}", f"captures_{captures_made}",
        f"opp_atari_{opp_atari}",
        f"caps_b_{board.captures[1]}", f"caps_w_{board.captures[-1]}",
        f"color_{'b' if color==1 else 'w'}",
    ]
    if captures_made > 0: toks.append("capturing_move")
    if opp_atari > 0:     toks.append("threatening_capture")
    if own_libs >= 4:      toks.append("safe_group")
    if own_libs <= 1:      toks.append("in_danger")
    if region == "center": toks.append("center_play")
    return " ".join(toks)


def _greedy_move(board, color):
    """
    Simple greedy Go bot:
      1. Capture if possible
      2. Put opponent in atari
      3. Play near own groups with few liberties (save)
      4. Center/influence heuristic
      5. Random legal move
    """
    S = GO_SIZE
    legal = board.legal_moves(color)
    if not legal: return None
    opp = -color

    best_score = -999; best_move = None

    for (r,c) in legal:
        sc = 0
        test = board.board.copy(); test[r,c] = color
        # Captures
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<S and 0<=nc<S and test[nr,nc]==opp:
                grp,libs = board._group_libs_on(test,nr,nc)
                if not libs: sc += len(grp)*10
        # Threaten opponent groups
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<S and 0<=nc<S and test[nr,nc]==opp:
                grp,libs = board._group_libs_on(test,nr,nc)
                if len(libs)==1: sc += 4
        # Own liberties
        grp,libs = board._group_libs_on(test,r,c)
        sc += len(libs)
        # Prefer center early
        center_dist = abs(r - S//2) + abs(c - S//2)
        sc -= center_dist * 0.3
        # Slight random tiebreak
        sc += RNG.uniform(0, 0.5)

        if sc > best_score:
            best_score = sc; best_move = (r,c)

    return best_move or RNG.choice(legal)


def go_play_game(black_fn, white_fn):
    """
    Play a complete 9×9 game between two move functions.
    Returns (board, move_log) where move_log = [(color, (r,c), move_num), ...]
    """
    board    = GoBoard()
    move_log = []
    consec_passes = 0

    for move_num in range(GO_MAX_MOVES):
        color = 1 if move_num % 2 == 0 else -1  # Black first
        fn    = black_fn if color == 1 else white_fn
        move  = fn(board, color)

        if move is None:
            consec_passes += 1
            if consec_passes >= 2: break
            continue
        consec_passes = 0

        r, c = move
        if board.is_legal(r, c, color):
            board.place(r, c, color)
            move_log.append((color, (r,c), move_num))
        else:
            # Illegal move — pass
            consec_passes += 1
            if consec_passes >= 2: break

    return board, move_log


def go_collect_training(n_games):
    pairs = []; t0 = time.time()
    print(f"  Playing {n_games} Go self-play training games (9×9)...")

    greedy_b = lambda board, color: _greedy_move(board, color)
    greedy_w = lambda board, color: _greedy_move(board, color)

    for gi in range(n_games):
        board, move_log = go_play_game(greedy_b, greedy_w)
        b_sc, w_sc, _, _ = board.score()
        # Credit: winner's moves are "best_move", loser's are "other_move"
        winner = 1 if b_sc > w_sc else -1

        # Replay to reconstruct boards at each move
        replay_board = GoBoard()
        for color, (r,c), move_num in move_log:
            # Territory-delta credit assignment: only label winner's move as best_move
            # if it actually improved their territory lead on that move
            before_b, before_w, _, _ = replay_board.score()

            if replay_board.is_legal(r, c, color):
                replay_board.place(r, c, color)

            after_b, after_w, _, _ = replay_board.score()

            if color == winner:
                improved = (color == 1 and (after_b - after_w) > (before_b - before_w)) or \
                           (color == -1 and (after_w - after_b) > (before_w - before_b))
                label = "best_move" if improved else "other_move"
            else:
                label = "other_move"

            feat_best = go_board_features(replay_board, (r,c), color, move_num)
            pairs.append((feat_best, label))

            # Negative examples: random other legal moves (only for confirmed best_moves)
            if label == "best_move":
                others = [m for m in replay_board.legal_moves(color) if m != (r,c)]
                neg_moves = RNG.sample(others, min(2, len(others)))
                for nm in neg_moves:
                    pairs.append((go_board_features(replay_board, nm, color, move_num),
                                  "other_move"))

        if (gi+1) % 50 == 0:
            elapsed = time.time()-t0
            print(f"    game {gi+1:>4}/{n_games}  pairs={len(pairs):>6,}  "
                  f"{elapsed:.0f}s  {(gi+1)/elapsed:.1f}g/s", end="\r")

    print()
    print(f"  Go training data: {len(pairs):,} examples from {n_games} games "
          f"({time.time()-t0:.1f}s)")
    return pairs


def go_pick_move(cypha, board, color, move_num):
    legal = board.legal_moves(color)
    if not legal: return None
    if len(legal) == 1: return legal[0]

    # Score all legal moves
    best_move = None; best_conf = -1.0
    # Limit candidates to avoid slowness on wide boards
    candidates = RNG.sample(legal, min(20, len(legal)))
    for m in candidates:
        feat = go_board_features(board, m, color, move_num)
        try:
            lbl, conf = cypha.infer(feat, verbose=False)
            if lbl == "best_move" and conf > best_conf:
                best_conf = conf; best_move = m
        except: pass
    if best_move is None:
        # Fall back to greedy
        best_move = _greedy_move(board, color)
    return best_move or RNG.choice(legal)


def go_run_test(cypha):
    wins = draws = losses = 0
    t0 = time.time()
    print(f"  Playing {N_GO_TEST} test Go games (Cypha Black vs greedy White)...")

    for i in range(N_GO_TEST):
        move_num_ref = [0]

        def cypha_black(board, color):
            mn = move_num_ref[0]; move_num_ref[0] += 1
            return go_pick_move(cypha, board, color, mn)

        def greedy_white(board, color):
            return _greedy_move(board, color)

        board, _ = go_play_game(cypha_black, greedy_white)
        b_sc, w_sc, b_terr, w_terr = board.score()

        if   b_sc > w_sc: wins+=1;   r="B+{:.1f}".format(b_sc-w_sc)
        elif w_sc > b_sc: losses+=1; r="W+{:.1f}".format(w_sc-b_sc)
        else:             draws+=1;  r="DRAW"

        print(f"    game {i+1:>3}/{N_GO_TEST}  {r}  "
              f"B={b_sc:.1f} W={w_sc:.1f}  "
              f"[W={wins} D={draws} L={losses}]  {time.time()-t0:.0f}s")

    return wins, draws, losses


# ─────────────────────────────────────────────────────────────────────────────
#  GENERIC TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_cypha(cypha, pairs, name):
    print(f"  Training Cypha-{name} on {len(pairs):,} examples...")
    RNG.shuffle(pairs)
    t0 = time.time()
    for i, (feat, label) in enumerate(pairs):
        cypha._cypha.train_step(feat, label)
        if (i+1) % 5000 == 0:
            print(f"    {i+1:>7,}/{len(pairs):,}  "
                  f"{(i+1)/(time.time()-t0):,.0f}/s", end="\r")
    elapsed = time.time()-t0
    print(f"\n  Training done: {elapsed:.1f}s  "
          f"({1000*elapsed/max(1,len(pairs)):.2f} ms/example)")


def probe_accuracy(cypha, pairs, name, n=2000):
    probe = RNG.sample(pairs, min(n, len(pairs)))
    correct = best_c = best_t = other_c = other_t = 0
    for feat, label in probe:
        try:
            pred, _ = cypha.infer(feat, verbose=False)
            if pred == label: correct += 1
            if label == "best_move" or label == "best_action":
                best_t += 1
                if pred == label: best_c += 1
            else:
                other_t += 1
                if pred == label: other_c += 1
        except: pass
    total = len(probe)
    acc   = 100.*correct/total if total else 0
    bpre  = 100.*best_c/best_t  if best_t  else 0
    opre  = 100.*other_c/other_t if other_t else 0
    print(f"  [{name}] Move accuracy: {acc:.1f}%  "
          f"(best_recall={bpre:.1f}%  other_recall={opre:.1f}%)")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    tmp_dir = tempfile.mkdtemp(prefix="cypha_games_")
    t_total = time.time()

    print("═"*72)
    print("  CYPHA HRNA — 3-GAME BENCHMARK  [Chess · Poker · Go]")
    print("═"*72)
    print(f"  Chess : {N_CHESS_TRAIN} train games (Stockfish depth {CHESS_TRAIN_DEPTH})  →  "
          f"{N_CHESS_TEST} test games vs Stockfish depth {CHESS_TEST_DEPTH}")
    print(f"  Poker : {N_POKER_TRAIN:,} train hands (MC iters={POKER_MC_ITERS})         →  "
          f"{N_POKER_TEST} test hands vs rule-based opponent")
    print(f"  Go    : {N_GO_TRAIN} train games (greedy self-play 9×9)     →  "
          f"{N_GO_TEST} test games vs greedy bot")
    print("═"*72)

    all_results = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  CHESS
    # ══════════════════════════════════════════════════════════════════════════
    if CHESS_OK:
        print("\n" + "─"*72)
        print("  ♟  CHESS")
        print("─"*72)
        cypha_chess = make_cypha(tmp_dir, "chess")
        pairs = chess_collect_training(N_CHESS_TRAIN)
        if pairs:
            RNG.shuffle(pairs)
            split = int(len(pairs)*0.85)
            train_cypha(cypha_chess, pairs[:split], "chess")
            chess_acc = probe_accuracy(cypha_chess, pairs[split:], "chess")

            print()
            cw, cd, cl, cresults = chess_run_test(cypha_chess)
            cn = cw+cd+cl
            chess_score = (cw + 0.5*cd)/max(cn,1)*100
            all_results["chess"] = {
                "wins":cw,"draws":cd,"losses":cl,
                "score_pct":chess_score,"move_acc":chess_acc,
                "results":" ".join(cresults),
            }
    else:
        print("\n  ♟  CHESS — SKIPPED (python-chess not installed)")

    # ══════════════════════════════════════════════════════════════════════════
    #  POKER
    # ══════════════════════════════════════════════════════════════════════════
    if POKER_OK:
        print("\n" + "─"*72)
        print("  ♠  POKER  (heads-up Texas Hold'em)")
        print("─"*72)
        cypha_poker = make_cypha(tmp_dir, "poker")
        pairs = poker_collect_training(N_POKER_TRAIN)
        if pairs:
            RNG.shuffle(pairs)
            split = int(len(pairs)*0.85)
            train_cypha(cypha_poker, pairs[:split], "poker")
            poker_acc = probe_accuracy(cypha_poker, pairs[split:], "poker")

            print()
            pw, pd, pl, p_net = poker_run_test(cypha_poker)
            pn = pw+pd+pl
            all_results["poker"] = {
                "wins":pw,"draws":pd,"losses":pl,
                "win_pct":100.*pw/max(pn,1),
                "net_chips":p_net,"move_acc":poker_acc,
            }
    else:
        print("\n  ♠  POKER — SKIPPED (treys not installed)")

    # ══════════════════════════════════════════════════════════════════════════
    #  GO
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*72)
    print("  ⚫  GO  (9×9, Chinese rules)")
    print("─"*72)
    cypha_go = make_cypha(tmp_dir, "go")
    pairs = go_collect_training(N_GO_TRAIN)
    if pairs:
        RNG.shuffle(pairs)
        split = int(len(pairs)*0.85)
        train_cypha(cypha_go, pairs[:split], "go")
        go_acc = probe_accuracy(cypha_go, pairs[split:], "go")

        print()
        gw, gd, gl = go_run_test(cypha_go)
        gn = gw+gd+gl
        all_results["go"] = {
            "wins":gw,"draws":gd,"losses":gl,
            "win_pct":100.*gw/max(gn,1),"move_acc":go_acc,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    total_t = time.time()-t_total
    print(f"\n\n{'═'*72}")
    print("  FINAL RESULTS — ALL GAMES")
    print("═"*72)

    if "chess" in all_results:
        r = all_results["chess"]
        n = r["wins"]+r["draws"]+r["losses"]
        print(f"\n  ♟  CHESS  (Cypha White vs Stockfish depth {CHESS_TEST_DEPTH} skill {CHESS_SKILL}/20)")
        print(f"     Move accuracy  : {r['move_acc']:.1f}%")
        print(f"     Games played   : {n}")
        print(f"     Wins / Draws / Losses  : {r['wins']} / {r['draws']} / {r['losses']}")
        print(f"     Score (W+½D)   : {r['wins']+0.5*r['draws']:.1f} / {n}  "
              f"({r['score_pct']:.1f}%)")
        print(f"     Results        : {r['results']}")

    if "poker" in all_results:
        r = all_results["poker"]
        n = r["wins"]+r["draws"]+r["losses"]
        print(f"\n  ♠  POKER  (Cypha vs rule-based opponent, {N_POKER_TEST} hands)")
        print(f"     Action accuracy: {r['move_acc']:.1f}%")
        print(f"     Hands played   : {n}")
        print(f"     Wins / Draws / Losses  : {r['wins']} / {r['draws']} / {r['losses']}")
        print(f"     Win rate       : {r['win_pct']:.1f}%")
        print(f"     Net chips      : {r['net_chips']:+.0f}")

    if "go" in all_results:
        r = all_results["go"]
        n = r["wins"]+r["draws"]+r["losses"]
        print(f"\n  ⚫  GO  (Cypha Black vs greedy White, 9×9 Chinese rules)")
        print(f"     Move accuracy  : {r['move_acc']:.1f}%")
        print(f"     Games played   : {n}")
        print(f"     Wins / Draws / Losses  : {r['wins']} / {r['draws']} / {r['losses']}")
        print(f"     Win rate       : {r['win_pct']:.1f}%")

    print(f"\n  Total wall time : {total_t:.1f}s")
    print("═"*72)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
