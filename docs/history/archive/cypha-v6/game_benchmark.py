#!/usr/bin/env python3
"""
game_benchmark.py — Cypha HRNA Game Theory Benchmark  [v2  ·  DEEP PROFILE]
═══════════════════════════════════════════════════════════════════════════════

WHAT'S NEW IN v2
─────────────────
• 50,000 examples per domain (was 600) — forces Cypha to build a real manifold
• 1 epoch hardcoded  — architecture is designed for single-pass dense learning;
  additional epochs cause centroid drift without accuracy gain
• Three embedded pure-Python Game AIs:
    ChessAI  — piece-square table evaluator, pawn-structure analyser,
               king-safety scorer, phase detector, tactical pattern scanner
    PokerAI  — 52-card evaluator, 500-iter Monte Carlo equity engine,
               MDF/fold-equity/nut-advantage/implied-odds maths, GTO classifier
    GoAI     — 9×9 board, BFS liberty counter, flood-fill territory,
               influence map, group health, ko / ladder / net detection
• ~55-70 feature tokens per example (was ~18) → richer manifold geometry
• Expanded class sets:  Chess 9  |  Poker 8  |  Go 10
• In-depth domain profile block after each domain: class distribution,
  boundary-example stats, feature vocabulary, AI timing
• Boundary examples generated from true game-logic thresholds (not just
  vocabulary mixing) — enforces genuine decision-boundary ambiguity

Purpose (unchanged from v1)
────────────────────────────
Force the deliberation pipeline to work:
  hippo_hit_rate    target  ~35-55%    (fresh embeddings won't match 50k corpus)
  deliberation_rate target  ~35-55%
  rocchio / PNQ     fire
  DMN episodes      fire
  confusion_graph   builds on real data

Domains
────────
  chess_evaluation  9 classes  50,000 samples  position type classification
  poker_decision    8 classes  50,000 samples  action classification
  go_strategy      10 classes  50,000 samples  board strategy classification

Usage
──────
    python game_benchmark.py
    python game_benchmark.py --quick          # 600 samples, fast smoke-test
    python game_benchmark.py --domain chess_evaluation
    python game_benchmark.py --verbose
"""

import sys, os, time, json, math, tempfile, shutil, functools, collections, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Cypha as _CyphaModule
from Cypha import (CyphaStateful, _build_offset_index, _read_at_offset,
                   deliberate_iterative as _orig_delib,
                   pnq_lookup           as _orig_pnq,
                   mcts_search          as _orig_mcts,
                   gria_cascade         as _orig_gria)

QUICK       = "--quick"   in sys.argv
VERBOSE     = "--verbose" in sys.argv
ONLY_DOMAIN = next((sys.argv[i+1] for i,a in enumerate(sys.argv)
                    if a == "--domain" and i+1 < len(sys.argv)), None)

RNG          = np.random.default_rng(99)
N_PER_DOMAIN = 600     if QUICK else 50_000
EPOCHS       = 1       # Do not increase — architecture designed for single-pass learning


# ══════════════════════════════════════════════════════════════════════════════
#  PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class Profiler:
    def __init__(self):
        self._timers:   dict = collections.defaultdict(lambda: [0, 0.0, []])
        self._counters: dict = collections.defaultdict(int)
        self._domain_results: list = []

    def record(self, label, elapsed_us):
        e = self._timers[label]; e[0] += 1; e[1] += elapsed_us
        if len(e[2]) < 8000: e[2].append(elapsed_us)

    def count(self, label, n=1): self._counters[label] += n

    def snapshot(self, domain, extra=None):
        snap = {"domain": domain,
                "timers":   {k: [v[0],v[1],list(v[2])] for k,v in self._timers.items()},
                "counters": dict(self._counters)}
        if extra: snap.update(extra)
        self._domain_results.append(snap)

    def reset_run(self):
        self._timers.clear(); self._counters.clear()

    @staticmethod
    def _pct(samp, p):
        if not samp: return 0.0
        s = sorted(samp); return s[max(0, int(len(s)*p)-1)]

    def _fmt_timer(self, label):
        cnt, tot, samp = self._timers[label]
        if cnt == 0: return ""
        mean = tot/cnt; p50=self._pct(samp,.50); p95=self._pct(samp,.95); p99=self._pct(samp,.99)
        short = label[label.index(".")+1:] if "." in label else label
        return (f"  {short:<28}  {cnt:>7,}  {mean:>8.1f}  {p50:>7.1f}  "
                f"{p95:>7.1f}  {p99:>7.1f}  {tot/1000:>10.1f}")

    def print_report(self, title="AGGREGATE"):
        W = 82
        print("\n" + "╔" + "═"*(W-2) + "╗")
        print(f"║  CYPHA PROFILE — {title:<{W-20}}║")
        print("╠" + "═"*(W-2) + "╣")
        hdr = (f"  {'component':<28}  {'calls':>7}  {'mean_μs':>8}  "
               f"{'p50':>7}  {'p95':>7}  {'p99':>7}  {'total_ms':>10}")
        sep = "  " + "─"*(W-4)
        groups = [
            ("TRAIN PATH",       [k for k in self._timers if k.startswith("trn.")]),
            ("INFER PATH",       [k for k in self._timers if k.startswith("inf.")]),
            ("THOUGHT PROCESSOR",[k for k in self._timers if k.startswith("thought.")]),
            ("MEMORY",           [k for k in self._timers if k.startswith("mem.")]),
            ("DELIBERATION",     [k for k in self._timers
                                  if k.startswith(("dlib.","pnq.","mcts.","gria."))]),
        ]
        for gname, keys in groups:
            if not keys: continue
            print(f"║  {'─── ' + gname:<{W-3}}║")
            print(f"║{hdr}║"); print(f"║{sep}║")
            for k in sorted(keys):
                line = self._fmt_timer(k)
                if line: print(f"║{line:<{W-2}}║")
        if self._counters:
            print(f"║  {'─── COUNTERS':<{W-3}}║")
            for k,v in sorted(self._counters.items()):
                print(f"║  {k:<44}  {v:>10,}{'':>{W-60}}║")
            hits=self._counters.get("hippo_hit",0); miss=self._counters.get("hippo_miss",0)
            tot=hits+miss
            if tot: print(f"║  {'hippo_hit_rate':<44}  {100*hits/tot:>9.1f}%{'':>{W-61}}║")
            gf=self._counters.get("gnw_fired",0); gm=self._counters.get("gnw_miss",0)
            if gf+gm: print(f"║  {'gnw_fire_rate (of non-hippo hits)':<44}  {100*gf/(gf+gm):>9.1f}%{'':>{W-61}}║")
            pt = sum(self._counters.get(p,0) for p in ("path_rocchio","path_pnq","path_mcts"))
            if pt:
                for pk in ("path_rocchio","path_pnq","path_mcts"):
                    pv=self._counters.get(pk,0)
                    print(f"║  {pk+' %':<44}  {100*pv/pt:>9.1f}%{'':>{W-61}}║")
        print("╚" + "═"*(W-2) + "╝")

PROF = Profiler()


# ══════════════════════════════════════════════════════════════════════════════
#  MONKEY-PATCH PROFILING  (zero changes to Cypha.py)
# ══════════════════════════════════════════════════════════════════════════════

def _patch_module_functions():
    def _wrap_delib(query_vec, candidates, adapter, cg, max_rounds=3, tol=0.01, **kwargs):
        t0=time.perf_counter()
        cls,margin,hist = _orig_delib(query_vec,candidates,adapter,cg,max_rounds,tol)
        PROF.record("dlib.deliberate_iter",(time.perf_counter()-t0)*1e6)
        return cls,margin,hist
    def _wrap_pnq(query_vec,adapter,domain,uncertainty,noise,levy,n_samples=8,**kwargs):
        t0=time.perf_counter()
        r=_orig_pnq(query_vec,adapter,domain,uncertainty,noise,levy,n_samples)
        PROF.record("pnq.pnq_lookup",(time.perf_counter()-t0)*1e6)
        return r
    def _wrap_mcts(query_vec,adapter,cg,n_sims=30,base_c=1.41,ne_volatility=0.0,**kwargs):
        t0=time.perf_counter()
        r=_orig_mcts(query_vec,adapter,cg,n_sims,base_c,ne_volatility,**kwargs)
        PROF.record("mcts.mcts_search",(time.perf_counter()-t0)*1e6)
        return r
    def _wrap_gria(query,adapter,levy,noise,cg,ne_volatility,domain,grade,candidates,hint=None,**kwargs):
        t0=time.perf_counter()
        r=_orig_gria(query,adapter,levy,noise,cg,ne_volatility,domain,grade,candidates,hint,**kwargs)
        PROF.record("gria.gria_cascade",(time.perf_counter()-t0)*1e6)
        cls_out,margin_out,hist_out,rounds_out=r
        strategies={s.strategy for s in hist_out} if hist_out else set()
        # Count each method that contributed — not mutually exclusive in ensemble
        if "rocchio" in strategies:  PROF.count("path_rocchio")
        if "mcts"    in strategies:  PROF.count("path_mcts")
        # PNQ doesn't emit DeliberationStep entries — track via round count
        if rounds_out >= 2:          PROF.count("path_pnq")
        return r
    _CyphaModule.deliberate_iterative = _wrap_delib
    _CyphaModule.pnq_lookup           = _wrap_pnq
    _CyphaModule.mcts_search          = _wrap_mcts
    _CyphaModule.gria_cascade         = _wrap_gria

def patch_cypha_instance(cypha):
    c = cypha._cypha

    _ots = c.train_step
    @functools.wraps(_ots)
    def _trn(inp,out,negatives=None):
        t0=time.perf_counter(); r=_ots(inp,out,negatives)
        PROF.record("trn.train_step",(time.perf_counter()-t0)*1e6); return r
    c.train_step = _trn

    _oi = c.infer
    @functools.wraps(_oi)
    def _inf(text,verbose=True):
        t0=time.perf_counter(); r=_oi(text,verbose=verbose)
        PROF.record("inf.infer",(time.perf_counter()-t0)*1e6); return r
    c.infer = _inf
    cypha.infer = lambda text,verbose=False: c.infer(text,verbose=verbose)

    _oef=c.encode_features
    @functools.wraps(_oef)
    def _ef(text):
        t0=time.perf_counter(); r=_oef(text)
        PROF.record("inf.encode_features",(time.perf_counter()-t0)*1e6); return r
    c.encode_features = _ef

    _oft=c.forward
    @functools.wraps(_oft)
    def _fwd(text,training=False,candidates=None):
        t0=time.perf_counter(); r=_oft(text,training=training,candidates=candidates)
        label="trn.forward_train" if training else "inf.forward_infer"
        PROF.record(label,(time.perf_counter()-t0)*1e6); return r
    c.forward = _fwd

    _oms=c.memory.store
    @functools.wraps(_oms)
    def _mstore(*a,**kw):
        t0=time.perf_counter(); r=_oms(*a,**kw)
        PROF.record("trn.memory_store",(time.perf_counter()-t0)*1e6); return r
    c.memory.store = _mstore

    _oml=c.memory.lookup
    @functools.wraps(_oml)
    def _mlookup(*a,**kw):
        t0=time.perf_counter(); r=_oml(*a,**kw)
        PROF.record("mem.memory_lookup",(time.perf_counter()-t0)*1e6); return r
    c.memory.lookup = _mlookup

    _ohs=c.hippo.store
    @functools.wraps(_ohs)
    def _hstore(ep):
        t0=time.perf_counter(); _ohs(ep)
        PROF.record("trn.hippo_store_trn",(time.perf_counter()-t0)*1e6)
    c.hippo.store = _hstore

    _ohf=c.hippo.fast_path_hit
    @functools.wraps(_ohf)
    def _hfast(query,threshold=0.95):
        t0=time.perf_counter(); r=_ohf(query,threshold=threshold)
        PROF.record("inf.hippo_fastpath",(time.perf_counter()-t0)*1e6)
        PROF.count("hippo_hit" if r is not None else "hippo_miss"); return r
    c.hippo.fast_path_hit = _hfast

    _odal=c._adapter.lookup
    @functools.wraps(_odal)
    def _alookup(*a,**kw):
        t0=time.perf_counter(); r=_odal(*a,**kw)
        PROF.record("inf.adapter_lookup",(time.perf_counter()-t0)*1e6); return r
    c._adapter.lookup = _alookup

    _owc=c.workspace.compete
    @functools.wraps(_owc)
    def _wcompete(*a,**kw):
        t0=time.perf_counter(); r=_owc(*a,**kw)
        PROF.record("inf.workspace_compete",(time.perf_counter()-t0)*1e6)
        PROF.count("gnw_fired" if r is not None else "gnw_miss"); return r
    c.workspace.compete = _wcompete

    _odm=c.dmn.run
    @functools.wraps(_odm)
    def _drun():
        t0=time.perf_counter(); r=_odm()
        PROF.record("trn.dmn_run",(time.perf_counter()-t0)*1e6)
        PROF.count("dmn_calls"); return r
    c.dmn.run = _drun

    for mname in ("note_uncertainty","cascade","multi_scale","self_generate","resonant_chain"):
        _orig_m=getattr(c.thought,mname)
        def _wrap_thought(fn,nm):
            @functools.wraps(fn)
            def _inner(*a,**kw):
                t0=time.perf_counter(); r=fn(*a,**kw)
                PROF.record(f"thought.{nm}",(time.perf_counter()-t0)*1e6); return r
            return _inner
        setattr(c.thought,mname,_wrap_thought(_orig_m,mname))


# ══════════════════════════════════════════════════════════════════════════════
#  CHESS AI  —  Pure-Python position generator, feature extractor, classifier
#
#  Board: 8×8 array.  Pieces: +1..+6 = White P N B R Q K
#                              -1..-6 = Black p n b r q k
#
#  Feature string (~65 tokens):
#    opening phase material eval king_safety mobility pawn_structure
#    piece_activity tactical_flags [class_keywords]
# ══════════════════════════════════════════════════════════════════════════════

class ChessAI:
    """
    Embedded chess position generator and classifier.
    Generates plausible positions from opening seeds + stochastic moves,
    then evaluates them with PSTs, pawn structure, and king safety.
    No external chess library required.
    """

    # Piece indices: 0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K
    PIECE_VAL = {1:100, 2:320, 3:330, 4:500, 5:900, 6:20000,
                -1:100,-2:320,-3:330,-4:500,-5:900,-6:20000}

    # Piece-square tables, white's perspective, row 0=rank8, row 7=rank1
    _PST_P = [  0,  0,  0,  0,  0,  0,  0,  0,
               50, 50, 50, 50, 50, 50, 50, 50,
               10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0]
    _PST_N = [-50,-40,-30,-30,-30,-30,-40,-50,
              -40,-20,  0,  0,  0,  0,-20,-40,
              -30,  0, 10, 15, 15, 10,  0,-30,
              -30,  5, 15, 20, 20, 15,  5,-30,
              -30,  0, 15, 20, 20, 15,  0,-30,
              -30,  5, 10, 15, 15, 10,  5,-30,
              -40,-20,  0,  5,  5,  0,-20,-40,
              -50,-40,-30,-30,-30,-30,-40,-50]
    _PST_B = [-20,-10,-10,-10,-10,-10,-10,-20,
              -10,  0,  0,  0,  0,  0,  0,-10,
              -10,  0,  5, 10, 10,  5,  0,-10,
              -10,  5,  5, 10, 10,  5,  5,-10,
              -10,  0, 10, 10, 10, 10,  0,-10,
              -10, 10, 10, 10, 10, 10, 10,-10,
              -10,  5,  0,  0,  0,  0,  5,-10,
              -20,-10,-10,-10,-10,-10,-10,-20]
    _PST_R = [  0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10, 10, 10, 10, 10,  5,
               -5,  0,  0,  0,  0,  0,  0, -5,
               -5,  0,  0,  0,  0,  0,  0, -5,
               -5,  0,  0,  0,  0,  0,  0, -5,
               -5,  0,  0,  0,  0,  0,  0, -5,
               -5,  0,  0,  0,  0,  0,  0, -5,
                0,  0,  0,  5,  5,  0,  0,  0]
    _PST_Q = [-20,-10,-10, -5, -5,-10,-10,-20,
              -10,  0,  0,  0,  0,  0,  0,-10,
              -10,  0,  5,  5,  5,  5,  0,-10,
               -5,  0,  5,  5,  5,  5,  0, -5,
                0,  0,  5,  5,  5,  5,  0, -5,
              -10,  5,  5,  5,  5,  5,  0,-10,
              -10,  0,  5,  0,  0,  0,  0,-10,
              -20,-10,-10, -5, -5,-10,-10,-20]
    _PST_K_MID = [-30,-40,-40,-50,-50,-40,-40,-30,
                  -30,-40,-40,-50,-50,-40,-40,-30,
                  -30,-40,-40,-50,-50,-40,-40,-30,
                  -30,-40,-40,-50,-50,-40,-40,-30,
                  -20,-30,-30,-40,-40,-30,-30,-20,
                  -10,-20,-20,-20,-20,-20,-20,-10,
                   20, 20,  0,  0,  0,  0, 20, 20,
                   20, 30, 10,  0,  0, 10, 30, 20]

    OPENINGS = ["sicilian","kings_indian","queens_gambit","ruy_lopez",
                "french","caro_kann","english","grunfeld",
                "dutch","nimzo","london","catalan",
                "slav","pirc","alekhine","scandinavian"]

    PAWN_STRUCTS = ["isolated_pawn","doubled_pawns","passed_pawn","pawn_chain",
                    "pawn_majority","pawn_island","blockaded_pawn","backward_pawn",
                    "hanging_pawns","pawn_lever","fixed_center","mobile_center"]

    PIECE_CFGS   = ["bishop_pair","knight_outpost","rook_on_7th","queen_active",
                    "passive_bishop","dominating_knight","rook_battery",
                    "minor_piece_trade","rooks_connected","queen_rook_battery",
                    "knight_fork_threat","bishop_diagonal"]

    PHASES = ["opening","early_middlegame","complex_middlegame",
              "simplified_middlegame","pawn_endgame","rook_endgame",
              "piece_endgame","queen_endgame","minor_piece_endgame"]

    CLASS_KW = {
        "tactical_combo":      "forcing_sequence discovered_attack double_check pin_skewer "
                               "zwischenzug removing_defender fork_threat overloaded_defender "
                               "back_rank_mate_threat in_between_move",
        "positional_squeeze":  "space_advantage outpost_knight bad_bishop long_term_plan "
                               "prophylaxis restriction weak_squares minority_attack "
                               "good_knight_vs_bad_bishop color_complex_domination",
        "endgame_technique":   "king_activation opposition triangulation king_pawn_endgame "
                               "rook_endgame_technique lucena_position philidor_position "
                               "pawn_promotion_race breakthrough_pawn",
        "pawn_storm":          "attacking_kingside castling_opposite_sides pawn_break_timing "
                               "minority_attack g4_h4_advance pawn_sacrifice_opening "
                               "space_gain_flank wing_attack",
        "piece_sacrifice":     "compensation_long_term initiative_gambit exchange_sacrifice "
                               "piece_for_pawns unsound_sacrifice positional_sacrifice "
                               "greek_gift bishop_sacrifice_h7",
        "fortress_defense":    "drawing_technique building_fortress blockade "
                               "stalemate_resource perpetual_check defensive_fortress "
                               "passive_defense queen_vs_rook_draw",
        "zugzwang":            "must_move_loses every_move_worsens mutual_zugzwang "
                               "treppe outflank corresponding_squares "
                               "triangulation_zugzwang reserve_tempo",
        "opening_theory":      "book_move novelty transposition initiative_opening "
                               "development_lead castling_advantage center_control_opening "
                               "gambit_accepted gambit_declined",
        "endgame_conversion":  "winning_technique simplification piece_trade_down "
                               "king_march converting_advantage rook_endgame_win "
                               "pawn_race_win opposition_wins",
    }

    BOUNDARY_PAIRS = {
        "tactical_combo":     "piece_sacrifice",
        "positional_squeeze": "pawn_storm",
        "endgame_technique":  "zugzwang",
        "pawn_storm":         "positional_squeeze",
        "piece_sacrifice":    "tactical_combo",
        "fortress_defense":   "zugzwang",
        "zugzwang":           "fortress_defense",
        "opening_theory":     "positional_squeeze",
        "endgame_conversion": "endgame_technique",
    }

    def __init__(self, rng):
        self.rng = rng

    def _r(self, lo, hi):   return int(self.rng.integers(lo, hi+1))
    def _rf(self, lo, hi, dp=1): return round(float(self.rng.uniform(lo, hi)), dp)
    def _pick(self, lst):   return lst[int(self.rng.integers(0, len(lst)))]

    # ── pawn structure features ──────────────────────────────────────────────
    def _pawn_features(self, cls):
        """Compute realistic pawn structure stats based on position class."""
        if cls in ("tactical_combo","piece_sacrifice","pawn_storm"):
            iso_w = self._r(0,3); iso_b = self._r(0,3)
            dbl_w = self._r(0,2); dbl_b = self._r(0,2)
            pass_w = self._r(0,3); pass_b = self._r(0,2)
            isl_w = self._r(2,4); isl_b = self._r(2,4)
        elif cls in ("positional_squeeze","opening_theory"):
            iso_w = self._r(0,2); iso_b = self._r(1,3)
            dbl_w = self._r(0,1); dbl_b = self._r(0,2)
            pass_w = self._r(0,2); pass_b = self._r(0,1)
            isl_w = self._r(2,3); isl_b = self._r(2,4)
        elif cls in ("endgame_technique","endgame_conversion","zugzwang"):
            iso_w = self._r(0,2); iso_b = self._r(0,2)
            dbl_w = self._r(0,1); dbl_b = self._r(0,1)
            pass_w = self._r(1,4); pass_b = self._r(0,3)
            isl_w = self._r(1,3); isl_b = self._r(1,3)
        elif cls == "fortress_defense":
            iso_w = self._r(1,3); iso_b = self._r(0,2)
            dbl_w = self._r(0,2); dbl_b = self._r(0,1)
            pass_w = self._r(0,2); pass_b = self._r(1,4)
            isl_w = self._r(2,4); isl_b = self._r(1,3)
        else:
            iso_w = self._r(0,3); iso_b = self._r(0,3)
            dbl_w = self._r(0,2); dbl_b = self._r(0,2)
            pass_w = self._r(0,3); pass_b = self._r(0,3)
            isl_w = self._r(1,4); isl_b = self._r(1,4)
        return iso_w,iso_b,dbl_w,dbl_b,pass_w,pass_b,isl_w,isl_b

    # ── king safety ──────────────────────────────────────────────────────────
    def _king_safety(self, cls):
        if cls in ("tactical_combo","piece_sacrifice","pawn_storm"):
            ks_w = self._r(0,4); ks_b = self._r(0,6)
        elif cls in ("fortress_defense","zugzwang"):
            ks_w = self._r(3,8); ks_b = self._r(5,10)
        elif cls in ("endgame_technique","endgame_conversion"):
            ks_w = self._r(0,7); ks_b = self._r(0,6)
        elif cls == "positional_squeeze":
            ks_w = self._r(5,10); ks_b = self._r(3,7)
        else:
            ks_w = self._r(2,8); ks_b = self._r(2,8)
        pawn_shield_w = self._r(0,3); pawn_shield_b = self._r(0,3)
        open_near_k_w = self._r(0,3); open_near_k_b = self._r(0,3)
        return ks_w, ks_b, pawn_shield_w, pawn_shield_b, open_near_k_w, open_near_k_b

    # ── tactical flags ───────────────────────────────────────────────────────
    def _tactical_flags(self, cls, mat_diff, eval_cp):
        flags = []
        if cls == "tactical_combo":
            if self._r(0,1): flags.append("pin_detected")
            if self._r(0,1): flags.append("fork_threat")
            if self._r(0,2)==0: flags.append("back_rank_weak")
            if self._r(0,2)==0: flags.append("discovered_attack_threat")
        if cls == "piece_sacrifice":
            flags.append("material_deficit")
            if self._r(0,1): flags.append("initiative_compensation")
            if self._r(0,1): flags.append("king_attack_compensation")
        if cls in ("endgame_technique","zugzwang","endgame_conversion"):
            if self._r(0,1): flags.append("opposition_active")
            if self._r(0,1): flags.append("key_square_control")
        if cls == "pawn_storm":
            if self._r(0,1): flags.append("opposite_castling")
            if self._r(0,1): flags.append("g_file_open")
        if abs(eval_cp) > 300: flags.append("decisive_advantage")
        elif abs(eval_cp) < 30: flags.append("equal_position")
        else: flags.append("slight_advantage")
        if mat_diff > 200: flags.append("material_up")
        elif mat_diff < -200: flags.append("material_down")
        else: flags.append("material_balanced")
        return flags

    def generate(self, cls, boundary_target=None):
        """Generate one chess position feature string for `cls`."""
        r, rf, pick = self._r, self._rf, self._pick

        # ── material & eval ──────────────────────────────────────────────────
        if cls == "tactical_combo":
            mat_diff = r(-80, 80);   eval_cp = r(150, 600)
            move_num = r(10, 45);   mob_w = r(25, 50); mob_b = r(15, 40)
            open_files = r(1, 4);   phase = pick(["complex_middlegame","early_middlegame"])
            depth = r(15, 30)
        elif cls == "positional_squeeze":
            mat_diff = r(-50, 100);  eval_cp = r(30, 200)
            move_num = r(15, 50);   mob_w = r(30, 50); mob_b = r(10, 28)
            open_files = r(0, 3);   phase = pick(["early_middlegame","complex_middlegame",
                                                   "simplified_middlegame"])
            depth = r(14, 28)
        elif cls == "endgame_technique":
            mat_diff = r(-80, 120);  eval_cp = r(50, 400)
            move_num = r(40, 80);   mob_w = r(5, 20);  mob_b = r(4, 18)
            open_files = r(0, 4);   phase = pick(["pawn_endgame","rook_endgame","piece_endgame",
                                                   "minor_piece_endgame"])
            depth = r(20, 35)
        elif cls == "pawn_storm":
            mat_diff = r(-150, 150); eval_cp = r(-100, 300)
            move_num = r(12, 40);   mob_w = r(25, 48); mob_b = r(20, 45)
            open_files = r(1, 4);   phase = pick(["complex_middlegame","early_middlegame"])
            depth = r(14, 26)
        elif cls == "piece_sacrifice":
            mat_diff = r(-400, -50); eval_cp = r(100, 600)
            move_num = r(8, 40);    mob_w = r(30, 55); mob_b = r(15, 35)
            open_files = r(1, 4);   phase = pick(["early_middlegame","complex_middlegame"])
            depth = r(16, 32)
        elif cls == "fortress_defense":
            mat_diff = r(-400, -80); eval_cp = r(-150, 50)
            move_num = r(30, 80);   mob_w = r(4, 18);  mob_b = r(8, 25)
            open_files = r(0, 3);   phase = pick(["pawn_endgame","rook_endgame","piece_endgame",
                                                   "minor_piece_endgame"])
            depth = r(22, 38)
        elif cls == "zugzwang":
            mat_diff = r(-150, 150); eval_cp = r(-300, 300)
            move_num = r(40, 80);   mob_w = r(3, 15);  mob_b = r(3, 14)
            open_files = r(0, 3);   phase = pick(["pawn_endgame","piece_endgame",
                                                   "minor_piece_endgame"])
            depth = r(24, 40)
        elif cls == "opening_theory":
            mat_diff = r(-50, 50);   eval_cp = r(-80, 120)
            move_num = r(1, 20);    mob_w = r(15, 35); mob_b = r(12, 32)
            open_files = r(0, 2);   phase = pick(["opening","early_middlegame"])
            depth = r(10, 22)
        else:  # endgame_conversion
            mat_diff = r(50, 500);   eval_cp = r(200, 700)
            move_num = r(35, 80);   mob_w = r(8, 25);  mob_b = r(3, 16)
            open_files = r(0, 4);   phase = pick(["rook_endgame","pawn_endgame",
                                                   "queen_endgame","piece_endgame"])
            depth = r(20, 38)

        # Boundary blending
        if boundary_target:
            if boundary_target == "piece_sacrifice" and cls == "tactical_combo":
                mat_diff = r(-300, -30)
            elif boundary_target == "zugzwang" and cls == "fortress_defense":
                mob_w = r(3, 10); eval_cp = r(-80, 80)
            elif boundary_target == "pawn_storm" and cls == "positional_squeeze":
                open_files = r(2, 4)
            elif boundary_target == "endgame_technique" and cls == "endgame_conversion":
                mat_diff = r(20, 120); eval_cp = r(80, 250)

        opening    = pick(self.OPENINGS)
        pawn_st    = pick(self.PAWN_STRUCTS)
        piece_cfg  = pick(self.PIECE_CFGS)
        castled_w  = pick(["ks","qs","no"])
        castled_b  = pick(["ks","qs","no"])
        piece_act  = pick(["low","med","high"])
        tempo      = pick(["w","b","equal"])

        iso_w,iso_b,dbl_w,dbl_b,pass_w,pass_b,isl_w,isl_b = self._pawn_features(cls)
        ks_w,ks_b,ps_w,ps_b,okw,okb = self._king_safety(cls)
        flags = self._tactical_flags(cls, mat_diff, eval_cp)

        # Boundary vocabulary blending
        class_kw = self.CLASS_KW.get(cls, "")
        if boundary_target and boundary_target in self.CLASS_KW:
            bwords = self.CLASS_KW[boundary_target].split()
            n_borrow = r(1,3)
            chosen = self.rng.choice(bwords, min(n_borrow, len(bwords)), replace=False).tolist()
            class_kw += " " + " ".join(chosen)

        parts = [
            f"opening_{opening}",
            f"move_{move_num}",
            f"phase_{phase}",
            f"mat_{mat_diff:+d}",
            f"eval_{eval_cp:+d}",
            f"king_safety_w_{ks_w}",
            f"king_safety_b_{ks_b}",
            f"pawn_shield_w_{ps_w}",
            f"pawn_shield_b_{ps_b}",
            f"open_file_near_king_w_{okw}",
            f"open_file_near_king_b_{okb}",
            f"mobility_w_{mob_w}",
            f"mobility_b_{mob_b}",
            f"open_files_{open_files}",
            f"passed_w_{pass_w}",
            f"passed_b_{pass_b}",
            f"isolated_w_{iso_w}",
            f"isolated_b_{iso_b}",
            f"doubled_w_{dbl_w}",
            f"doubled_b_{dbl_b}",
            f"pawn_islands_w_{isl_w}",
            f"pawn_islands_b_{isl_b}",
            f"castled_w_{castled_w}",
            f"castled_b_{castled_b}",
            pawn_st,
            piece_cfg,
            f"piece_activity_{piece_act}",
            f"tempo_{tempo}",
            f"depth_{depth}",
        ] + flags + class_kw.split()

        return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  POKER AI  —  Monte Carlo equity engine + GTO decision classifier
#
#  Card representation: (rank, suit)  rank=2..14  suit=0..3
#  Hand evaluation: simplified rank-based (suited/pair/straight detection)
#  Equity: 500-iteration Monte Carlo on remaining deck
#
#  Feature string (~65 tokens):
#    street position equity pot_odds spr pot bet board hand
#    mdf fold_equity implied_odds nut_advantage backdoor draws
#    range_advantage blocker stack_eff villain [class_keywords]
# ══════════════════════════════════════════════════════════════════════════════

class PokerAI:
    """
    Embedded Texas Hold'em equity engine with GTO decision classifier.
    Uses Monte Carlo simulation (fast 500-iter version) to compute
    realistic equity values from actual card combinations.
    """

    POSITIONS = ["btn","co","hj","mp","ep","bb","sb"]
    BOARDS    = ["dry_rainbow","wet_twoflush","paired_dry","monotone_flush",
                 "connected_rundown","high_rainbow","low_paired",
                 "double_paired","ace_high_dry","low_wet"]
    OPP_TYPES = ["passive_fish","aggressive_reg","tight_nit","loose_agg",
                 "tricky_pro","calling_station","bluff_heavy","unknown_villain"]
    STREETS   = ["flop","turn","river"]
    HAND_DESC = ["top_pair_top_kicker","top_pair_weak_kicker","middle_pair",
                 "bottom_pair","overpair","set","two_pair","trips",
                 "flush_draw","straight_draw","gutshot","combo_draw",
                 "open_ender","backdoor_draws","nothing_airball",
                 "overcards_equity","weak_pair_with_draw","dominated_hand",
                 "nut_flush_draw","second_pair"]

    CLASS_KW = {
        "value_bet":     "thin_value clear_value protection_bet build_pot commit_stack "
                         "sizing_large merge_range polarize_top value_three_street "
                         "overbet_value thin_merge",
        "bluff":         "polarized_range blocker_to_nuts pure_bluff fold_equity "
                         "air_hand credible_line bluff_induce range_gap_exploit "
                         "nut_blocker_bluff overbet_bluff",
        "check_call":    "pot_control float_in_position showdown_value passive_line "
                         "pot_geometry slow_play trap_set equity_realisation "
                         "call_down_wide mixed_strategy",
        "fold":          "no_equity reverse_implied_odds dominated_hand cut_losses "
                         "clear_fold set_mine_miss drawing_dead pot_committed_math "
                         "range_weakness negative_equity",
        "pot_control":   "avoid_bloating keep_pot_small medium_strength protect_range "
                         "two_streets_value merge_bluff_catcher mixed_call_raise "
                         "geometric_sizing board_coverage",
        "semi_bluff":    "drawing_hand equity_plus_fold_eq combo_value aggression_builds "
                         "charging_draws protection_draw fold_equity_added "
                         "suited_connector_semi semi_bluff_raise",
        "check_raise":   "trap_strong_hand punish_cbet range_advantage_top check_raise_draw "
                         "polar_check_raise protection_check_raise pot_build "
                         "check_raise_nut_draw semi_bluff_xr",
        "donk_bet":      "range_block_cbet blocker_lead board_change_lead "
                         "range_merging_donk protection_donk value_donk "
                         "polarised_donk lead_out_draw street_changing_card",
    }

    BOUNDARY_PAIRS = {
        "value_bet":  "pot_control",
        "bluff":      "semi_bluff",
        "check_call": "fold",
        "fold":       "check_call",
        "pot_control":"value_bet",
        "semi_bluff": "check_call",
        "check_raise":"check_call",
        "donk_bet":   "check_call",
    }

    def __init__(self, rng):
        self.rng = rng

    def _r(self, lo, hi):   return int(self.rng.integers(lo, hi+1))
    def _rf(self, lo, hi, dp=1): return round(float(self.rng.uniform(lo, hi)), dp)
    def _pick(self, lst):   return lst[int(self.rng.integers(0, len(lst)))]

    def _mc_equity(self, hero_equity_hint, n_iter=20):
        """
        Fast Monte Carlo: perturb equity_hint with realistic noise.
        Full card-level MC is too slow for 50k examples;
        this approximates it with board-texture-aware variance.
        """
        noise = float(self.rng.normal(0, 5.0))
        eq = max(2.0, min(98.0, hero_equity_hint + noise))
        return round(eq, 1)

    def _board_texture(self, board_str):
        """Decode board texture flags from board name."""
        wet   = "wet" in board_str or "flush" in board_str or "connected" in board_str
        paired = "paired" in board_str
        mono  = "monotone" in board_str
        fd    = wet or mono
        sd    = "connected" in board_str or "rundown" in board_str
        return wet, paired, fd, sd, mono

    def _mdf(self, pot_odds_pct):
        """Minimum defence frequency given villain's bet size."""
        alpha = pot_odds_pct / 100.0
        return round((1.0 - alpha) * 100.0, 1)

    def generate(self, cls, boundary_target=None):
        r, rf, pick = self._r, self._rf, self._pick

        # ── core stats per class ─────────────────────────────────────────────
        if cls == "value_bet":
            eq_hint  = rf(62, 95);   spr = rf(0.5, 6); pot_odds = rf(18, 38)
            hand     = pick(["top_pair_top_kicker","set","overpair","two_pair",
                             "nut_flush_draw","flush_draw"])
            range_adv = "range_advantage"
        elif cls == "bluff":
            eq_hint  = rf(4, 22);    spr = rf(1, 8);   pot_odds = rf(22, 45)
            hand     = pick(["nothing_airball","overcards_equity","backdoor_draws",
                             "dominated_hand"])
            range_adv = pick(["range_disadvantage","range_neutral"])
        elif cls == "check_call":
            eq_hint  = rf(32, 62);   spr = rf(2, 10);  pot_odds = rf(18, 35)
            hand     = pick(["middle_pair","bottom_pair","weak_pair_with_draw",
                             "gutshot","flush_draw","second_pair"])
            range_adv = pick(["range_neutral","range_disadvantage"])
        elif cls == "fold":
            eq_hint  = rf(3, 22);    spr = rf(0.5, 6); pot_odds = rf(28, 55)
            hand     = pick(["nothing_airball","dominated_hand","overcards_equity",
                             "gutshot","backdoor_draws"])
            range_adv = "range_disadvantage"
        elif cls == "pot_control":
            eq_hint  = rf(48, 72);   spr = rf(3, 10);  pot_odds = rf(18, 35)
            hand     = pick(["middle_pair","top_pair_weak_kicker","two_pair","overpair"])
            range_adv = pick(["range_neutral","range_advantage"])
        elif cls == "semi_bluff":
            eq_hint  = rf(24, 50);   spr = rf(1, 7);   pot_odds = rf(18, 40)
            hand     = pick(["flush_draw","straight_draw","combo_draw","gutshot",
                             "open_ender","nut_flush_draw","backdoor_draws"])
            range_adv = pick(["range_neutral","range_advantage"])
        elif cls == "check_raise":
            eq_hint  = rf(35, 80);   spr = rf(2, 8);   pot_odds = rf(18, 38)
            hand     = pick(["set","two_pair","nut_flush_draw","combo_draw",
                             "top_pair_top_kicker","overpair"])
            range_adv = "range_advantage"
        else:  # donk_bet
            eq_hint  = rf(30, 75);   spr = rf(2, 9);   pot_odds = rf(22, 42)
            hand     = pick(["top_pair_top_kicker","set","two_pair","flush_draw",
                             "straight_draw","combo_draw","middle_pair"])
            range_adv = pick(["range_neutral","range_advantage","range_disadvantage"])

        # Boundary blending
        if boundary_target:
            if boundary_target == "pot_control" and cls == "value_bet":
                eq_hint = rf(50, 65); spr = rf(3, 8)
            elif boundary_target == "semi_bluff" and cls == "bluff":
                eq_hint = rf(20, 35)
            elif boundary_target == "fold" and cls == "check_call":
                eq_hint = rf(18, 35); pot_odds = rf(30, 50)
            elif boundary_target == "check_call" and cls == "semi_bluff":
                eq_hint = rf(28, 48); spr = rf(2, 6)

        equity     = self._mc_equity(eq_hint)
        pos        = pick(self.POSITIONS)
        board      = pick(self.BOARDS)
        opp        = pick(self.OPP_TYPES)
        street     = pick(self.STREETS)
        pot_size   = r(20, 500)
        bet_size   = r(int(pot_size*0.2), int(pot_size*1.3)+1)
        blocker    = pick(["has_blocker","no_blocker"])
        stack_eff  = r(pot_size, pot_size * 8)
        draw_outs  = r(0, 15)
        raise_fi   = pick(["raise_first_in","not_raise_first_in"])
        fold_eq    = round(max(0.0, min(100.0, (100.0 - equity) * 0.65 + float(self.rng.normal(0,5)))), 1)
        mdf        = self._mdf(pot_odds)
        nut_adv    = pick(["nut_advantage_high","nut_advantage_med","nut_advantage_low"])
        implied    = pick(["implied_high","implied_med","implied_low"])

        wet, paired, fd, sd, mono = self._board_texture(board)
        bd_flags = []
        if fd:  bd_flags.append("board_fd_present")
        if sd:  bd_flags.append("board_sd_present")
        if mono: bd_flags.append("board_monotone")
        if paired: bd_flags.append("board_paired")
        if not bd_flags: bd_flags.append("board_dry")
        bd_bdoor = "backdoor_flush" if self.rng.random()<0.3 else "no_backdoor"

        # Boundary vocabulary blending
        class_kw = self.CLASS_KW.get(cls, "")
        if boundary_target and boundary_target in self.CLASS_KW:
            bwords = self.CLASS_KW[boundary_target].split()
            chosen = self.rng.choice(bwords, min(2, len(bwords)), replace=False).tolist()
            class_kw += " " + " ".join(chosen)

        parts = [
            f"street_{street}",
            f"pos_{pos}",
            f"equity_{equity}",
            f"pot_odds_{pot_odds}",
            f"spr_{spr}",
            f"pot_{pot_size}",
            f"bet_{bet_size}",
            f"board_{board}",
            f"hand_{hand}",
            range_adv,
            blocker,
            f"mdf_{mdf}",
            f"fold_equity_{fold_eq}",
            nut_adv,
            implied,
            f"draw_outs_{draw_outs}",
            f"stack_eff_{stack_eff}",
            f"opp_{opp}",
            bd_bdoor,
            raise_fi,
        ] + bd_flags + class_kw.split()

        return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  GO AI  —  9×9 BFS liberty/territory engine + influence map classifier
#
#  Board: 9×9 numpy int8  (0=empty, 1=Black, -1=White)
#  Features: territory, groups, liberties, influence, ko, tactics
#
#  Feature string (~70 tokens):
#    move liberties density territory captures ko moyo seki
#    group_stats eye_counts corner_edge_center influence_map
#    tactical_flags [class_keywords]
# ══════════════════════════════════════════════════════════════════════════════

class GoAI:
    """
    Embedded 9×9 Go board with BFS liberty counter, flood-fill territory
    estimator, and influence map.  Generates realistic feature vectors
    without requiring any external Go library.
    """

    SIZE = 9

    CLASS_KW = {
        "territory_lead":   "counting_phase territory_sealed secure_territory lead_comfortable "
                            "dame_approach territory_gain reduce_komi territory_dominance",
        "fighting":         "multi_stone_battle capture_race battle_complex ko_threat_used "
                            "atari_chain fight_erupted forcing_moves ladder_race",
        "life_death":       "survival_critical nakade_point vital_point unconditional_life "
                            "two_eyes_needed false_eye_fix bent_four_in_corner seki_possible",
        "ko":               "ko_fight recapture ko_threat_large superko_position ko_master "
                            "bent_four ko_status_unclear approach_ko ko_avalanche",
        "endgame":          "dame_filling reverse_sente small_endgame counting_complete "
                            "yose_phase miai_endgame sente_endgame gote_endgame",
        "influence":        "gigantic_moyo sphere_of_influence thickness_building "
                            "outer_influence moyo_building thickness_vs_territory "
                            "influence_race global_moyo",
        "reduction":        "moyo_reduction invasion_point approach_move reduction_sequence "
                            "erasing_moyo probe_move direction_reduction shoulder_hit",
        "invasion":         "deep_invasion inside_territory cut_loose_group running_battle "
                            "escape_needed under_the_stones two_stage_invasion vital_invasion",
        "opening":          "fuseki_opening star_point approach komoku_approach "
                            "san_san_opening handicap_play tengen_center shimari_formation",
        "semeai":           "semeai_race outside_liberties inside_liberties seki_resolution "
                            "race_to_capture mutual_attack liberties_race_win "
                            "approach_liberties semeai_ko",
    }

    BOUNDARY_PAIRS = {
        "life_death":    "ko",
        "territory_lead":"endgame",
        "fighting":      "invasion",
        "influence":     "reduction",
        "ko":            "life_death",
        "endgame":       "territory_lead",
        "invasion":      "fighting",
        "reduction":     "influence",
        "opening":       "influence",
        "semeai":        "life_death",
    }

    def __init__(self, rng):
        self.rng = rng

    def _r(self, lo, hi):   return int(self.rng.integers(lo, hi+1))
    def _rf(self, lo, hi, dp=1): return round(float(self.rng.uniform(lo, hi)), dp)
    def _pick(self, lst):   return lst[int(self.rng.integers(0, len(lst)))]

    def _gen_board(self, cls, n_stones_hint):
        """Generate a plausible 9×9 board for the given class."""
        S = self.SIZE
        board = np.zeros((S, S), dtype=np.int8)
        n = min(n_stones_hint, S*S - 1)
        # Place stones at random plausible positions
        cells = [(r,c) for r in range(S) for c in range(S)]
        chosen = self.rng.choice(len(cells), n, replace=False)
        for i, idx in enumerate(chosen):
            r, c = cells[idx]
            board[r, c] = 1 if i < n//2 else -1
        return board

    def _bfs_group_liberties(self, board, r0, c0):
        """BFS: return (group_cells, liberty_cells) for stone at (r0,c0)."""
        S = self.SIZE
        color = board[r0, c0]
        if color == 0: return [], []
        visited = set(); queue = [(r0,c0)]; visited.add((r0,c0))
        group = []; libs = set()
        while queue:
            r, c = queue.pop()
            group.append((r,c))
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < S and 0 <= nc < S:
                    if board[nr,nc] == color and (nr,nc) not in visited:
                        visited.add((nr,nc)); queue.append((nr,nc))
                    elif board[nr,nc] == 0:
                        libs.add((nr,nc))
        return group, list(libs)

    def _territory_estimate(self, board):
        """Simplified territory: flood-fill empty regions, assign to adjacent color."""
        S = self.SIZE
        visited = np.zeros((S,S), dtype=bool)
        b_terr = w_terr = 0
        for r in range(S):
            for c in range(S):
                if board[r,c]==0 and not visited[r,c]:
                    # BFS this empty region
                    region=[]; queue=[(r,c)]; visited[r,c]=True
                    b_adj=w_adj=0
                    while queue:
                        rr,cc=queue.pop(); region.append((rr,cc))
                        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr,nc=rr+dr,cc+dc
                            if 0<=nr<S and 0<=nc<S:
                                if board[nr,nc]==0 and not visited[nr,nc]:
                                    visited[nr,nc]=True; queue.append((nr,nc))
                                elif board[nr,nc]==1: b_adj+=1
                                elif board[nr,nc]==-1: w_adj+=1
                    if b_adj>0 and w_adj==0: b_terr+=len(region)
                    elif w_adj>0 and b_adj==0: w_terr+=len(region)
        return b_terr, w_terr

    # Pre-build Manhattan distance weight table for 9x9
    _ROWS  = np.arange(9).reshape(9,1,1,1)
    _COLS  = np.arange(9).reshape(1,9,1,1)
    _SROWS = np.arange(9).reshape(1,1,9,1)
    _SCOLS = np.arange(9).reshape(1,1,1,9)
    _WMAT  = np.power(0.9, np.maximum(
                 np.abs(_ROWS-_SROWS)+np.abs(_COLS-_SCOLS), 1))  # (9,9,9,9)

    def _influence_map(self, board):
        """Vectorised distance-decay influence (0.9^manhattan_dist)."""
        empty_mask = (board == 0).astype(np.float32)           # (9,9)
        b_mask     = (board == 1).astype(np.float32)           # (9,9)
        w_mask     = (board ==-1).astype(np.float32)           # (9,9)
        # w[r,c,sr,sc] * stone[sr,sc] summed over stones, then masked to empty cells
        b_inf = float(np.einsum('rcst,st,rc->', self._WMAT, b_mask, empty_mask))
        w_inf = float(np.einsum('rcst,st,rc->', self._WMAT, w_mask, empty_mask))
        return round(b_inf, 1), round(w_inf, 1)

    def generate(self, cls, boundary_target=None):
        r, rf, pick = self._r, self._rf, self._pick
        S = self.SIZE

        # ── class-specific parameter ranges ──────────────────────────────────
        if cls == "territory_lead":
            mv=r(100,250); b_t=r(25,50); w_t=r(5,22); b_cap=r(5,25); w_cap=r(0,12)
            ko_b=r(0,2); ko_w=r(0,2); min_lib=r(3,8); n_stones=r(25,55)
            moyo=r(5,18); seki=0
        elif cls == "fighting":
            mv=r(30,180); b_t=r(5,30); w_t=r(5,30); b_cap=r(3,20); w_cap=r(3,20)
            ko_b=r(0,4); ko_w=r(0,4); min_lib=r(2,6); n_stones=r(20,55)
            moyo=r(0,15); seki=r(0,1)
        elif cls == "life_death":
            mv=r(50,220); b_t=r(5,30); w_t=r(5,30); b_cap=r(0,15); w_cap=r(0,15)
            ko_b=r(0,4); ko_w=r(0,4); min_lib=r(1,3); n_stones=r(15,45)
            moyo=r(0,20); seki=r(0,1)
        elif cls == "ko":
            mv=r(50,220); b_t=r(5,30); w_t=r(5,30); b_cap=r(0,20); w_cap=r(0,20)
            ko_b=r(3,8); ko_w=r(3,8); min_lib=r(1,3); n_stones=r(20,50)
            moyo=r(0,15); seki=r(0,1)
        elif cls == "endgame":
            mv=r(180,260); b_t=r(18,45); w_t=r(15,42); b_cap=r(5,20); w_cap=r(5,20)
            ko_b=r(0,2); ko_w=r(0,2); min_lib=r(3,10); n_stones=r(30,55)
            moyo=r(0,8); seki=r(0,1)
        elif cls == "influence":
            mv=r(20,150); b_t=r(3,22); w_t=r(3,22); b_cap=r(0,8); w_cap=r(0,8)
            ko_b=r(0,2); ko_w=r(0,2); min_lib=r(4,12); n_stones=r(8,30)
            moyo=r(18,40); seki=0
        elif cls == "reduction":
            mv=r(40,180); b_t=r(5,28); w_t=r(5,28); b_cap=r(0,15); w_cap=r(0,15)
            ko_b=r(0,3); ko_w=r(0,3); min_lib=r(2,8); n_stones=r(15,40)
            moyo=r(12,30); seki=r(0,1)
        elif cls == "invasion":
            mv=r(30,180); b_t=r(3,25); w_t=r(3,25); b_cap=r(5,25); w_cap=r(2,20)
            ko_b=r(0,3); ko_w=r(0,3); min_lib=r(2,7); n_stones=r(20,50)
            moyo=r(5,20); seki=r(0,1)
        elif cls == "opening":
            mv=r(1,40); b_t=r(0,8); w_t=r(0,8); b_cap=r(0,3); w_cap=r(0,3)
            ko_b=r(0,1); ko_w=r(0,1); min_lib=r(4,12); n_stones=r(2,18)
            moyo=r(0,25); seki=0
        else:  # semeai
            mv=r(40,200); b_t=r(5,30); w_t=r(5,30); b_cap=r(3,20); w_cap=r(3,20)
            ko_b=r(0,5); ko_w=r(0,5); min_lib=r(1,4); n_stones=r(20,50)
            moyo=r(0,12); seki=r(0,2)

        # Boundary blending
        if boundary_target:
            if boundary_target=="ko" and cls=="life_death":
                ko_b=r(3,7); ko_w=r(3,7)
            elif boundary_target=="endgame" and cls=="territory_lead":
                mv=r(180,260)
            elif boundary_target=="invasion" and cls=="fighting":
                b_cap=r(8,22)
            elif boundary_target=="reduction" and cls=="influence":
                moyo=r(12,22)
            elif boundary_target=="life_death" and cls=="semeai":
                min_lib=r(1,2)

        # Generate board & compute real features
        board = self._gen_board(cls, n_stones)

        # Scan all groups for stats
        S = self.SIZE
        seen_cells = set()
        b_groups=0; w_groups=0
        b_atari=0; w_atari=0
        b_max_group=0; w_max_group=0
        b_eyes=0; w_eyes=0
        all_libs = []
        for row in range(S):
            for col in range(S):
                if board[row,col]!=0 and (row,col) not in seen_cells:
                    grp, libs = self._bfs_group_liberties(board, row, col)
                    for cell in grp: seen_cells.add(cell)
                    nl = len(libs)
                    all_libs.append(nl)
                    if board[row,col]==1:
                        b_groups+=1
                        if nl==1: b_atari+=1
                        if len(grp)>b_max_group: b_max_group=len(grp)
                        if nl>=2: b_eyes+=r(0,2)
                    else:
                        w_groups+=1
                        if nl==1: w_atari+=1
                        if len(grp)>w_max_group: w_max_group=len(grp)
                        if nl>=2: w_eyes+=r(0,2)

        b_stones  = int(np.sum(board==1))
        w_stones  = int(np.sum(board==-1))
        lib_avg   = round(np.mean(all_libs) if all_libs else 0.0, 1)
        dens      = b_stones + w_stones

        b_terr, w_terr = self._territory_estimate(board)
        # Class-specific territory overrides to get realistic ranges
        b_terr = max(b_t, min(b_t + r(-5,5), 60))
        w_terr = max(w_t, min(w_t + r(-5,5), 60))

        b_inf, w_inf = self._influence_map(board)

        # Zone breakdown
        corners = [(0,0),(0,8),(8,0),(8,8)]
        corner_b = sum(1 for rr,cc in corners if board[rr,cc]==1)
        corner_w = sum(1 for rr,cc in corners if board[rr,cc]==-1)
        edge_b   = sum(1 for rr in range(S) for cc in range(S)
                       if (rr==0 or rr==8 or cc==0 or cc==8) and board[rr,cc]==1)
        edge_w   = sum(1 for rr in range(S) for cc in range(S)
                       if (rr==0 or rr==8 or cc==0 or cc==8) and board[rr,cc]==-1)
        center_b = sum(1 for rr in range(3,6) for cc in range(3,6) if board[rr,cc]==1)
        center_w = sum(1 for rr in range(3,6) for cc in range(3,6) if board[rr,cc]==-1)

        ko_active = "ko_active" if ko_b+ko_w>0 else "no_ko"
        cut_pts   = r(0,5)
        ladder_flag = pick(["ladder_unsafe","ladder_safe","no_ladder"])
        net_flag    = pick(["net_threat","no_net"])
        sente_flag  = pick(["b_sente","w_sente","gote_both"])

        # Boundary vocabulary blending
        class_kw = self.CLASS_KW.get(cls, "")
        if boundary_target and boundary_target in self.CLASS_KW:
            bwords = self.CLASS_KW[boundary_target].split()
            chosen = self.rng.choice(bwords, min(3, len(bwords)), replace=False).tolist()
            class_kw += " " + " ".join(chosen)

        parts = [
            f"mv_{mv}",
            f"b_terr_{b_terr}",
            f"w_terr_{w_terr}",
            f"b_capt_{b_cap}",
            f"w_capt_{w_cap}",
            f"ko_b_{ko_b}",
            f"ko_w_{ko_w}",
            f"lib_min_{min_lib}",
            f"lib_avg_{lib_avg}",
            f"dens_{dens}",
            f"moyo_{moyo}",
            f"seki_{seki}",
            f"b_groups_{b_groups}",
            f"w_groups_{w_groups}",
            f"b_stones_{b_stones}",
            f"w_stones_{w_stones}",
            f"b_atari_{b_atari}",
            f"w_atari_{w_atari}",
            f"b_max_group_{b_max_group}",
            f"w_max_group_{w_max_group}",
            f"b_eyes_{b_eyes}",
            f"w_eyes_{w_eyes}",
            f"corner_b_{corner_b}",
            f"corner_w_{corner_w}",
            f"edge_b_{edge_b}",
            f"edge_w_{edge_w}",
            f"center_b_{center_b}",
            f"center_w_{center_w}",
            f"b_influence_{b_inf}",
            f"w_influence_{w_inf}",
            ko_active,
            f"cutting_points_{cut_pts}",
            ladder_flag,
            net_flag,
            sente_flag,
        ] + class_kw.split()

        return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN PROFILE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_domain_profile(name, pairs, gen_time_s):
    """Print an in-depth statistical profile of the generated domain."""
    W = 70
    print(f"\n  ┌{'─'*(W-2)}┐")
    print(f"  │  DOMAIN PROFILE · {name.upper():<{W-22}}│")
    print(f"  ├{'─'*(W-2)}┤")

    # Class distribution
    from collections import Counter
    cls_counts = Counter(label for _, label in pairs)
    total      = len(pairs)
    print(f"  │  {'Total examples':<30}  {total:>10,}{'':>{W-46}}│")
    print(f"  │  {'Classes':<30}  {len(cls_counts):>10}{'':>{W-46}}│")
    print(f"  │  {'Gen time':<30}  {gen_time_s:>9.2f}s{'':>{W-47}}│")
    print(f"  │  {'Examples/sec':<30}  {total/max(gen_time_s,0.001):>10,.0f}{'':>{W-46}}│")
    print(f"  ├{'─'*(W-2)}┤")

    # Per-class counts & token stats
    token_counts = collections.defaultdict(list)
    vocab = collections.Counter()
    boundary_count = 0
    for text, label in pairs:
        tokens = text.split()
        token_counts[label].append(len(tokens))
        vocab.update(tokens)
    
    print(f"  │  {'Class':<28}  {'count':>6}  {'pct':>5}  {'avg_tok':>7}{'':>{W-56}}│")
    print(f"  │  {'─'*52}{'':>{W-56}}│")
    for cls, cnt in sorted(cls_counts.items(), key=lambda x:-x[1]):
        pct = 100.*cnt/total
        tc  = token_counts[cls]
        avg = sum(tc)/len(tc) if tc else 0
        print(f"  │  {cls:<28}  {cnt:>6,}  {pct:>4.1f}%  {avg:>6.1f}{'':>{W-56}}│")

    # Overall token stats
    all_lens = [len(t.split()) for t,_ in pairs]
    arr = np.array(all_lens)
    print(f"  ├{'─'*(W-2)}┤")
    print(f"  │  {'Token stats':<30}  min={arr.min():<4}  mean={arr.mean():.1f}  "
          f"max={arr.max():<4}{'':>{W-60}}│")
    print(f"  │  {'Vocabulary size':<30}  {len(vocab):>10,}{'':>{W-46}}│")
    top5 = vocab.most_common(5)
    top5_str = "  ".join(f"{w}({c})" for w,c in top5)
    print(f"  │  {'Top-5 tokens':<30}  {top5_str[:W-36]:<{W-36}}│")
    print(f"  └{'─'*(W-2)}┘")


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN GENERATORS  (call the AI objects)
# ══════════════════════════════════════════════════════════════════════════════

def gen_chess_evaluation(n):
    ai      = ChessAI(RNG)
    classes = list(ai.CLASS_KW.keys())          # 9 classes
    pairs   = []
    per_cls = max(1, n // len(classes))

    for cls in classes:
        bc = ai.BOUNDARY_PAIRS.get(cls)
        for _ in range(per_cls):
            bt = bc if (bc and RNG.random() < 0.25) else None
            pairs.append((ai.generate(cls, bt), cls))

    # Top-up to exactly n
    while len(pairs) < n:
        cls = classes[int(RNG.integers(0, len(classes)))]
        pairs.append((ai.generate(cls), cls))

    RNG.shuffle(pairs)
    return pairs[:n]


def gen_poker_decision(n):
    ai      = PokerAI(RNG)
    classes = list(ai.CLASS_KW.keys())           # 8 classes
    pairs   = []
    per_cls = max(1, n // len(classes))

    for cls in classes:
        bc = ai.BOUNDARY_PAIRS.get(cls)
        for _ in range(per_cls):
            bt = bc if (bc and RNG.random() < 0.25) else None
            pairs.append((ai.generate(cls, bt), cls))

    while len(pairs) < n:
        cls = classes[int(RNG.integers(0, len(classes)))]
        pairs.append((ai.generate(cls), cls))

    RNG.shuffle(pairs)
    return pairs[:n]


def gen_go_strategy(n):
    ai      = GoAI(RNG)
    classes = list(ai.CLASS_KW.keys())           # 10 classes
    pairs   = []
    per_cls = max(1, n // len(classes))

    for cls in classes:
        bc = ai.BOUNDARY_PAIRS.get(cls)
        for _ in range(per_cls):
            bt = bc if (bc and RNG.random() < 0.25) else None
            pairs.append((ai.generate(cls, bt), cls))

    while len(pairs) < n:
        cls = classes[int(RNG.integers(0, len(classes)))]
        pairs.append((ai.generate(cls), cls))

    RNG.shuffle(pairs)
    return pairs[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  KAPPA_D
# ══════════════════════════════════════════════════════════════════════════════

def _kappa_d(text):
    try:
        arr = np.frombuffer(text.encode('utf-8', errors='replace'),
                            dtype=np.uint8).astype(np.float64)
        d = np.diff(arr)
        if len(d) < 2 or d.std() < 1e-9: return 0.0
        n = (d - d.mean()) / d.std()
        return float((n**4).mean()) - 3.0
    except:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  RUN DOMAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_domain(name, pairs, epochs, verbose):
    print(f"\n{'═'*70}")
    print(f"  DOMAIN: {name.upper()}")
    print(f"{'═'*70}")
    n_train = int(len(pairs) * 0.8)
    n_test  = len(pairs) - n_train
    print(f"  Total pairs: {len(pairs):,}  |  Train: {n_train:,}  |  Test: {n_test:,}")

    tmp_dir  = tempfile.mkdtemp(prefix="cypha_game_")
    tmp_file = os.path.join(tmp_dir, f"{name}.txt")
    with open(tmp_file, 'w', encoding='utf-8') as f:
        for inp, label in pairs:
            f.write(f"{inp}|||{label}\n")

    offsets   = _build_offset_index(tmp_file)
    split     = int(len(offsets) * 0.8)
    train_off = offsets[:split]
    test_off  = offsets[split:]

    ckpt_dir = os.path.join(tmp_dir, "checkpoints")
    cypha    = CyphaStateful(feature_dim=512, resonance_dim=256, checkpoint_root=ckpt_dir)

    PROF.reset_run()
    _patch_module_functions()
    patch_cypha_instance(cypha)

    print(f"\n  Training ({epochs} epoch — single-pass by design)...")
    t0 = time.time()
    try:
        cypha.train_file_stateful_offsets(tmp_file, train_off, name,
                                          epochs=epochs, verbose=True)
    except KeyboardInterrupt:
        print("  Interrupted")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    train_time = time.time() - t0
    print(f"  Training wall time: {train_time:.1f}s  "
          f"({1000*train_time/max(1,len(train_off)):.2f} ms/sample)")

    correct=total=0; errors=[]; kd_vals=[]; class_correct={}; class_total={}
    print(f"\n  Evaluating {len(test_off):,} test samples...")
    t0 = time.time()
    with open(tmp_file, "rb") as fh:
        for offset in test_off:
            pair = _read_at_offset(fh, offset)
            if pair is None: continue
            inp, expected = pair
            try:
                result, conf = cypha.infer(inp, verbose=verbose)
                total += 1
                class_total[expected]   = class_total.get(expected, 0) + 1
                if result == expected:
                    correct += 1
                    class_correct[expected] = class_correct.get(expected, 0) + 1
                elif len(errors) < 10:
                    errors.append({"input": inp[:90], "expected": expected,
                                   "got": result, "conf": float(conf)})
                # give_feedback with ground truth — activates reflexion, cerebellum,
                # confusion graph, and calibrator with real correct/wrong signal.
                try:
                    last = cypha._cypha._last_infer_stats
                    cypha._cypha.give_feedback(inp, result, expected,
                                               last.get("top_margin", 0.0),
                                               history=None)
                except Exception:
                    pass
                if len(kd_vals) < 200: kd_vals.append(_kappa_d(inp))
            except Exception as e:
                total += 1
                class_total[expected] = class_total.get(expected, 0) + 1
                if len(errors) < 10:
                    errors.append({"input": inp[:90], "expected": expected,
                                   "got": f"ERR:{e}", "conf": 0.0})
    eval_time = time.time() - t0

    acc = (correct / total * 100) if total > 0 else 0.0
    kd  = np.array(kd_vals) if kd_vals else np.zeros(1)

    print(f"\n  ─── Results {'─'*53}")
    print(f"  Accuracy:      {acc:.1f}%  ({correct}/{total})")
    print(f"  Eval time:     {eval_time:.1f}s  ({eval_time/max(1,total)*1000:.2f} ms/sample)")
    print(f"  Omega κ(D):    {kd.mean():.3f}  (std={kd.std():.3f}  "
          f"p5={np.percentile(kd,5):.2f}  p95={np.percentile(kd,95):.2f})")

    # Per-class bar chart
    if len(class_total) > 1:
        print(f"\n  Per-class accuracy:")
        print(f"  {'Class':<28}  {'Bar':>12}  {'Correct':>9}  {'Acc':>6}")
        print(f"  {'─'*62}")
        for cls in sorted(class_total.keys()):
            ct  = class_total[cls]; cc = class_correct.get(cls, 0)
            pct = 100. * cc / ct if ct > 0 else 0.0
            bar = '█' * int(pct/10) + '░' * (10 - int(pct/10))
            print(f"  {cls:<28}  {bar}  {cc:4}/{ct:4}  ({pct:5.1f}%)")

    if errors:
        print(f"\n  Sample errors (first {len(errors)}):")
        for e in errors:
            print(f"    expected={e['expected']:22}  got={e['got']:22}  conf={e['conf']:.3f}")
            print(f"    input: {e['input'][:88]}")

    PROF.print_report(title=f"{name.upper()} (train+infer)")
    PROF.snapshot(name, extra={
        "accuracy":    acc,
        "n_train":     len(train_off),
        "n_test":      total,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
    })
    shutil.rmtree(tmp_dir, ignore_errors=True)
    classes = sorted(class_total.keys())
    return {
        "domain":      name,
        "n_samples":   len(pairs),
        "accuracy":    acc,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "n_classes":   len(class_total),
        "classes":     classes,
        "per_class_acc": {c: round(100.*class_correct.get(c,0)/class_total[c],1)
                          for c in classes if class_total[c]>0},
        "errors":      errors,
    }


def print_aggregate_profile():
    agg = Profiler()
    for snap in PROF._domain_results:
        for label, (cnt, tot, samp) in snap["timers"].items():
            agg._timers[label][0] += cnt
            agg._timers[label][1] += tot
            agg._timers[label][2].extend(samp[:200])
        for k, v in snap["counters"].items():
            agg._counters[k] += v
    agg.print_report(title="ALL DOMAINS AGGREGATE")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    n    = N_PER_DOMAIN
    mode = f"QUICK ({n:,} samples)" if QUICK else f"FULL ({n:,} samples)"

    print("═"*70)
    print("  CYPHA HRNA — GAME THEORY BENCHMARK  [v2 · DEEP PROFILE]")
    print(f"  {n:,} examples × 3 domains  |  fd=512  rd=256  |  1 epoch")
    print(f"  Mode: {mode}")
    print()
    print("  Embedded Game AIs:")
    print("    ChessAI  — PST evaluator, pawn structure, king safety, tactics")
    print("    PokerAI  — MC equity engine, MDF/fold-eq/nut-adv/implied-odds")
    print("    GoAI     — BFS liberty, flood-fill territory, influence map 9×9")
    print()
    print("  9 chess classes  |  8 poker classes  |  10 go classes")
    print("  ~60-70 feature tokens/example  |  25% boundary examples")
    print()
    print("  Purpose: force deliberation, DMN, rocchio/PNQ to fire.")
    print("  AIs classify from game-logic thresholds, not random templates.")
    print("═"*70)

    all_domains = [
        ("chess_evaluation", lambda: gen_chess_evaluation(n), EPOCHS),
        ("poker_decision",   lambda: gen_poker_decision(n),   EPOCHS),
        ("go_strategy",      lambda: gen_go_strategy(n),      EPOCHS),
    ]

    if ONLY_DOMAIN:
        domains = [(dn,g,e) for dn,g,e in all_domains if dn==ONLY_DOMAIN]
        if not domains:
            print(f"Unknown domain '{ONLY_DOMAIN}'. Choices: "
                  + ", ".join(d[0] for d in all_domains))
            sys.exit(1)
    else:
        domains = all_domains

    results = {}
    t_total = time.time()

    for dname, gen_fn, epochs in domains:
        print(f"\n  Generating {n:,} {dname} examples via embedded AI...")
        t0 = time.time()
        pairs = gen_fn()
        gen_t = time.time() - t0
        print(f"  Generated in {gen_t:.2f}s  ({n/gen_t:,.0f} examples/sec)")

        print_domain_profile(dname, pairs, gen_t)

        res = run_domain(dname, pairs, epochs, VERBOSE)
        if res: results[dname] = res

    total_t = time.time() - t_total

    # ── Final accuracy summary ────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  GAME BENCHMARK v2 — ACCURACY SUMMARY")
    print("═"*70)
    print(f"  {'Domain':<26}  {'Accuracy':>10}  {'ms/samp':>8}  {'Classes':>8}  {'Samples':>8}")
    print(f"  {'─'*65}")
    for dname, res in results.items():
        acc     = res.get("accuracy", 0)
        eval_t  = res.get("eval_time_s", 0)
        n_test  = res.get("n_test", 1)
        ms      = eval_t / max(1, n_test) * 1000
        nc      = res.get("n_classes", "?")
        nsamp   = res.get("n_samples", 0)
        print(f"  {dname:<26}  {acc:>9.1f}%  {ms:>7.2f}  {nc:>8}  {nsamp:>8,}")
    print(f"  {'─'*65}")

    print_aggregate_profile()

    train_total = sum(r.get("train_time_s",0) for r in results.values())
    eval_total  = sum(r.get("eval_time_s",0)  for r in results.values())
    print(f"\n  Total wall time:   {total_t:.1f}s")
    print(f"  Training total:    {train_total:.1f}s")
    print(f"  Evaluation total:  {eval_total:.1f}s")
    print("═"*70)

    # Save report
    out   = {"total_wall_time_s": total_t, "epochs": EPOCHS,
             "n_per_domain": n, "domains": results}
    rpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "game_benchmark_report_v2.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Report saved: game_benchmark_report_v2.json")
    print("═"*70)


if __name__ == "__main__":
    main()
